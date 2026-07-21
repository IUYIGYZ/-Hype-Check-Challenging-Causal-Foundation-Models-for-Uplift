#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_benchmark.data import prepare_data
from tune_baselines import (
    _cleanup_model,
    _fit_and_evaluate,
    _sample_params,
    _select_best,
    _write_json,
    _write_trials,
)


MODEL_NAME = "dragonnet"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_gpu_ids(value: str) -> list[str]:
    result = []
    for raw in value.split(","):
        gpu = raw.strip()
        if gpu and gpu not in result:
            result.append(gpu)
    if not result:
        raise ValueError("At least one GPU ID is required")
    return result


def _parse_trial_indices(value: str) -> list[int]:
    result = []
    seen = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        trial = int(item)
        if trial < 0:
            raise ValueError("Trial indices must be non-negative")
        if trial in seen:
            raise ValueError(f"Duplicate trial index {trial}")
        seen.add(trial)
        result.append(trial)
    if not result:
        raise ValueError("At least one trial index is required")
    return sorted(result)


def _read_config(source_dir: Path) -> dict:
    path = source_dir / "tuning_config.json"
    if not path.exists():
        raise ValueError(f"Missing tuning config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    critical = config.get("critical_settings", {})
    required = {
        "dataset",
        "cleaned_root",
        "max_rows_requested",
        "seed",
        "search_seed",
        "val_fraction",
        "test_fraction",
        "objective",
        "neural_max_epochs",
    }
    missing = sorted(required - set(critical))
    if missing:
        raise ValueError(f"Source tuning config is missing settings: {missing}")
    if critical["dataset"] != "criteo":
        raise ValueError(
            f"This continuation script is for Criteo, got {critical['dataset']!r}"
        )
    return config


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"Missing trial metrics: {path}")
    return pd.read_csv(path).to_dict(orient="records")


def _successful_dragonnet_trials(rows: list[dict]) -> set[int]:
    completed = set()
    for row in rows:
        if str(row.get("model")) != MODEL_NAME:
            continue
        if str(row.get("status")) != "ok":
            continue
        value = row.get("trial")
        if pd.notna(value):
            completed.add(int(value))
    return completed


def _resolve_target(config: dict, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--total-trials must be at least 1")
        return requested
    target = config.get("trial_targets", {}).get(MODEL_NAME)
    if target is None:
        raise ValueError(
            "No DragonNet trial target in source config; pass --total-trials"
        )
    return int(target)


def _partition(trials: list[int], gpu_ids: list[str]) -> dict[str, list[int]]:
    return {
        gpu: trials[index::len(gpu_ids)]
        for index, gpu in enumerate(gpu_ids)
        if trials[index::len(gpu_ids)]
    }


def _worker_command(
    script: Path,
    source_dir: Path,
    worker_dir: Path,
    gpu_id: str,
    trials: list[int],
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(script),
        "--worker",
        "--source-dir",
        str(source_dir),
        "--worker-dir",
        str(worker_dir),
        "--gpu-id",
        gpu_id,
        "--trial-indices",
        ",".join(str(value) for value in trials),
    ]


def _write_status(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = _now()
    _write_json(path, payload)


def _worker_main(args) -> int:
    source_dir = args.source_dir.expanduser().resolve()
    worker_dir = args.worker_dir.expanduser().resolve()
    worker_dir.mkdir(parents=True, exist_ok=False)
    config = _read_config(source_dir)
    critical = config["critical_settings"]
    trials = _parse_trial_indices(args.trial_indices)

    worker_config = {
        "started_at": _now(),
        "source_dir": str(source_dir),
        "gpu_id_physical": args.gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trial_indices": trials,
        "critical_settings": critical,
    }
    _write_json(worker_dir / "worker_config.json", worker_config)

    max_rows_requested = int(critical["max_rows_requested"])
    max_rows = None if max_rows_requested == 0 else max_rows_requested
    print(
        f"[{_now()}] GPU {args.gpu_id}: preparing "
        f"{'ALL cleaned rows' if max_rows is None else max_rows} rows",
        flush=True,
    )
    data = prepare_data(
        cleaned_root=Path(critical["cleaned_root"]),
        dataset=critical["dataset"],
        outcome=critical.get("outcome_requested"),
        max_rows=max_rows,
        seed=int(critical["seed"]),
        val_fraction=float(critical["val_fraction"]),
        test_fraction=float(critical["test_fraction"]),
    )

    rows: list[dict] = []
    trial_path = worker_dir / "trial_metrics.csv"
    for trial in trials:
        params = _sample_params(
            MODEL_NAME,
            trial,
            search_seed=int(critical["search_seed"]),
            neural_max_epochs=int(critical["neural_max_epochs"]),
        )
        row: dict[str, object] = {
            "dataset": data.dataset,
            "outcome": data.outcome,
            "model": MODEL_NAME,
            "trial": trial,
            "status": "error",
            "objective": critical["objective"],
            "params_json": json.dumps(params, sort_keys=True),
            "fit_seconds": float("nan"),
            "predict_seconds": float("nan"),
            "error": "",
        }
        model = None
        print(
            f"[{_now()}] GPU {args.gpu_id}: START trial {trial} params={params}",
            flush=True,
        )
        try:
            (
                model,
                _,
                _,
                _,
                _,
                metrics,
                fit_seconds,
                predict_seconds,
            ) = _fit_and_evaluate(
                MODEL_NAME,
                params,
                data,
                evaluation_split="validation",
                model_seed=int(critical["seed"]),
                device="cuda",
            )
            objective_value = float(metrics[critical["objective"]])
            if not np.isfinite(objective_value):
                raise RuntimeError(
                    f"Validation objective {critical['objective']} is NaN or Inf"
                )
            row.update(
                {
                    "status": "ok",
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    **metrics,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _cleanup_model(MODEL_NAME, model)
            model = None

        rows.append(row)
        _write_trials(trial_path, rows)
        print(
            f"[{_now()}] GPU {args.gpu_id}: END trial {trial} "
            f"status={row['status']} objective={row.get(critical['objective'])} "
            f"error={row['error']}",
            flush=True,
        )

    successful = _successful_dragonnet_trials(rows)
    summary = {
        "finished_at": _now(),
        "gpu_id_physical": args.gpu_id,
        "requested_trials": trials,
        "successful_trials": sorted(successful),
        "failed_trials": sorted(set(trials) - successful),
    }
    _write_json(worker_dir / "worker_summary.json", summary)
    return 0


def _merge_results(
    source_dir: Path,
    worker_dirs: list[Path],
    config: dict,
    total_trials: int,
    run_dir: Path,
) -> tuple[list[int], list[int]]:
    source_path = source_dir / "trial_metrics.csv"
    original_rows = _read_rows(source_path)
    all_rows = list(original_rows)
    for worker_dir in worker_dirs:
        worker_path = worker_dir / "trial_metrics.csv"
        if worker_path.exists():
            all_rows.extend(_read_rows(worker_path))

    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise RuntimeError("No trial rows available for merging")
    frame["_ok_priority"] = (frame["status"].astype(str) != "ok").astype(int)
    frame["_source_order"] = np.arange(len(frame))
    frame = frame.sort_values(
        ["model", "trial", "_ok_priority", "_source_order"],
        kind="mergesort",
    )
    frame = frame.drop_duplicates(["model", "trial"], keep="first")
    frame = frame.drop(columns=["_ok_priority", "_source_order"])
    frame = frame.sort_values(["model", "trial"], kind="mergesort")
    merged_rows = frame.to_dict(orient="records")

    backup = source_dir / "trial_metrics.before_dual_gpu.csv"
    if not backup.exists():
        shutil.copy2(source_path, backup)
    _write_trials(source_path, merged_rows)
    _write_trials(run_dir / "merged_trial_metrics.csv", merged_rows)

    critical = config["critical_settings"]
    models = list(critical.get("models", []))
    if not models:
        models = sorted(frame["model"].astype(str).unique().tolist())
    best = _select_best(
        merged_rows,
        objective=str(critical["objective"]),
        models=models,
    )
    _write_json(source_dir / "best_params.json", best)
    _write_json(run_dir / "merged_best_params.json", best)

    successful = _successful_dragonnet_trials(merged_rows)
    expected = set(range(total_trials))
    return sorted(expected & successful), sorted(expected - successful)


def _detach(args, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    child_args = [value for value in sys.argv[1:] if value != "--detach"]
    if not any(
        value == "--run-dir" or value.startswith("--run-dir=")
        for value in child_args
    ):
        child_args.extend(["--run-dir", str(run_dir)])
    log_path = run_dir / "supervisor.log"
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), *child_args],
            cwd=Path(__file__).parent,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    (run_dir / "supervisor.pid").write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    print(f"Detached dual-GPU supervisor PID: {process.pid}")
    print(f"Run directory: {run_dir}")
    print(f"Supervisor log: {log_path}")
    return 0


def _supervisor_main(args) -> int:
    source_dir = args.source_dir.expanduser().resolve()
    config = _read_config(source_dir)
    total_trials = _resolve_target(config, args.total_trials)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    source_rows = _read_rows(source_dir / "trial_metrics.csv")
    completed = _successful_dragonnet_trials(source_rows)
    missing = sorted(set(range(total_trials)) - completed)
    assignments = _partition(missing, gpu_ids)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else source_dir / f"dual_gpu_dragonnet_{stamp}"
    )
    script = Path(__file__).resolve()
    commands = {}
    for index, (gpu_id, trials) in enumerate(assignments.items()):
        worker_dir = run_dir / f"worker_{index}_gpu_{gpu_id}"
        commands[gpu_id] = _worker_command(
            script, source_dir, worker_dir, gpu_id, trials
        )

    print(f"Already successful DragonNet trials: {sorted(completed)}")
    print(f"Missing DragonNet trials: {missing}")
    print(f"GPU assignments: {assignments}")
    if not missing:
        print("All requested DragonNet trials are already complete.")
        return 0
    if args.dry_run:
        for gpu_id, command in commands.items():
            print(
                f"GPU {gpu_id}: "
                + subprocess.list2cmdline(command)
            )
        return 0
    if args.detach:
        return _detach(args, run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    status_path = run_dir / "status.json"
    started_at = _now()
    status = {
        "state": "running",
        "started_at": started_at,
        "source_dir": str(source_dir),
        "total_trials": total_trials,
        "completed_before_start": sorted(completed),
        "assignments": assignments,
        "workers": {},
    }
    _write_status(status_path, status)

    workers = {}
    thread_count = str(args.cpu_threads_per_worker)
    for index, (gpu_id, trials) in enumerate(assignments.items()):
        worker_dir = run_dir / f"worker_{index}_gpu_{gpu_id}"
        log_path = logs_dir / f"gpu_{gpu_id}.log"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
            }
        )
        stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            commands[gpu_id],
            cwd=Path(__file__).parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=False,
            close_fds=True,
        )
        workers[gpu_id] = {
            "process": process,
            "stream": stream,
            "worker_dir": worker_dir,
            "log_path": log_path,
            "trials": trials,
        }
        status["workers"][gpu_id] = {
            "pid": process.pid,
            "trials": trials,
            "log": str(log_path),
            "worker_dir": str(worker_dir),
            "state": "running",
        }
        print(
            f"[{_now()}] START GPU {gpu_id} pid={process.pid} trials={trials}",
            flush=True,
        )
    _write_status(status_path, status)

    while True:
        running = False
        for gpu_id, worker in workers.items():
            code = worker["process"].poll()
            worker_state = status["workers"][gpu_id]
            if code is None:
                running = True
                worker_state["state"] = "running"
            else:
                worker_state["state"] = "finished" if code == 0 else "failed"
                worker_state["exit_code"] = code
        _write_status(status_path, status)
        if not running:
            break
        time.sleep(args.poll_seconds)

    for worker in workers.values():
        worker["stream"].close()

    worker_dirs = [worker["worker_dir"] for worker in workers.values()]
    successful, still_missing = _merge_results(
        source_dir,
        worker_dirs,
        config,
        total_trials,
        run_dir,
    )
    status["successful_trials_after_merge"] = successful
    status["missing_trials_after_merge"] = still_missing
    status["state"] = "completed" if not still_missing else "incomplete"
    status["finished_at"] = _now()
    _write_status(status_path, status)
    print(
        f"[{_now()}] MERGED successful={successful} missing={still_missing}",
        flush=True,
    )
    return 0 if not still_missing else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue missing Criteo DragonNet trials on disjoint GPUs and "
            "atomically merge results into the original tuning directory."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--total-trials", type=int, default=None)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trial-indices", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cpu_threads_per_worker < 1:
        raise ValueError("--cpu-threads-per-worker must be at least 1")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.worker:
        if args.worker_dir is None or args.gpu_id is None or args.trial_indices is None:
            raise ValueError(
                "Worker mode requires --worker-dir, --gpu-id, and --trial-indices"
            )
        return _worker_main(args)
    return _supervisor_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
