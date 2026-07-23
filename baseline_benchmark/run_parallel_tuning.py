#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SUPPORTED_DATASETS = ("retailhero", "hillstrom", "lzd", "criteo")
DEFAULT_MODELS = "t_learner,x_learner,dr_learner,dragonnet,causalpfn"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run independent full-data tuning jobs in parallel. Each dataset gets "
            "its own log and result directory. Use --detach to survive SSH logout."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(SUPPORTED_DATASETS),
        help="Comma-separated datasets.",
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--exclusive-datasets",
        default="criteo",
        help="Comma-separated datasets that must run alone; Criteo is exclusive by default.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="Physical GPU IDs assigned one per concurrent job.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--cpu-threads-per-job", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--traditional-trials", type=int, default=20)
    parser.add_argument("--dragonnet-trials", type=int, default=20)
    parser.add_argument("--neural-max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument(
        "--objective",
        choices=[
            "qini_auc_normalized",
            "qini_coefficient",
            "uplift_auc_normalized",
            "auuc",
            "uplift_at_10pct",
            "uplift_at_20pct",
        ],
        default="qini_auc_normalized",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "data_A_cleaned",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).parent / "parallel_tuning_runs",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact batch directory. Normally generated automatically.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start the supervisor in a new session and return immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print child commands without starting tuning jobs.",
    )
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    result = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in result:
            result.append(item)
    return result


def _validate_args(args):
    datasets = _parse_csv(args.datasets)
    if not datasets:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(datasets) - set(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown datasets {unknown}; supported={list(SUPPORTED_DATASETS)}"
        )

    requested_exclusive = set(_parse_csv(args.exclusive_datasets))
    unknown_exclusive = sorted(requested_exclusive - set(SUPPORTED_DATASETS))
    if unknown_exclusive:
        raise ValueError(
            "Unknown exclusive datasets: "
            f"{unknown_exclusive}"
        )
    exclusive = requested_exclusive & set(datasets)

    gpu_ids = _parse_csv(args.gpu_ids)
    if args.max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    if args.max_rows < 0:
        raise ValueError("max_rows must be 0 or a positive integer")
    for name in (
        "traditional_trials",
        "dragonnet_trials",
        "neural_max_epochs",
        "cpu_threads_per_job",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")
    if args.device != "cpu":
        if not gpu_ids:
            raise ValueError("CUDA/auto mode requires at least one --gpu-ids value")
        if args.max_parallel > len(gpu_ids):
            raise ValueError(
                "max_parallel cannot exceed GPU IDs in CUDA/auto mode"
            )
    return datasets, exclusive, gpu_ids


def _new_run_dir(args) -> Path:
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.run_root / f"batch_{stamp}").expanduser().resolve()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _detach(args, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    child_args = [value for value in sys.argv[1:] if value != "--detach"]
    has_run_dir = any(
        value == "--run-dir" or value.startswith("--run-dir=")
        for value in child_args
    )
    if not has_run_dir:
        child_args.extend(["--run-dir", str(run_dir)])

    supervisor_log = run_dir / "supervisor.log"
    with supervisor_log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), *child_args],
            cwd=Path(__file__).parent,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    (run_dir / "supervisor.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Detached supervisor PID: {process.pid}")
    print(f"Run directory: {run_dir}")
    print(f"Supervisor log: {supervisor_log}")
    print(f"Status file: {run_dir / 'status.json'}")


def _tuning_command(args, dataset: str, results_root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).parent / "tune_baselines.py"),
        "--dataset",
        dataset,
        "--models",
        args.models,
        "--cleaned-root",
        str(args.cleaned_root.expanduser().resolve()),
        "--output-root",
        str(results_root),
        "--max-rows",
        str(args.max_rows),
        "--traditional-trials",
        str(args.traditional_trials),
        "--dragonnet-trials",
        str(args.dragonnet_trials),
        "--neural-max-epochs",
        str(args.neural_max_epochs),
        "--seed",
        str(args.seed),
        "--search-seed",
        str(args.search_seed),
        "--objective",
        args.objective,
        "--val-fraction",
        str(args.val_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--device",
        args.device,
    ]


def _write_status(
    path: Path,
    *,
    started_at: str,
    run_dir: Path,
    args,
    datasets: list[str],
    exclusive: set[str],
    jobs: dict[str, dict],
    state: str,
) -> None:
    _atomic_json(
        path,
        {
            "state": state,
            "started_at": started_at,
            "updated_at": _now(),
            "run_dir": str(run_dir),
            "full_cleaned_data": args.max_rows == 0,
            "max_rows": args.max_rows,
            "max_parallel": args.max_parallel,
            "exclusive_datasets": sorted(exclusive),
            "datasets": datasets,
            "jobs": jobs,
        },
    )


def _run_supervisor(args, run_dir: Path, datasets, exclusive, gpu_ids) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    results_root = run_dir / "results"
    logs_dir.mkdir(exist_ok=True)
    results_root.mkdir(exist_ok=True)
    started_at = _now()
    status_path = run_dir / "status.json"

    commands = {
        dataset: _tuning_command(args, dataset, results_root)
        for dataset in datasets
    }
    config = {
        "started_at": started_at,
        "python": sys.executable,
        "datasets": datasets,
        "models": args.models,
        "max_parallel": args.max_parallel,
        "exclusive_datasets": sorted(exclusive),
        "gpu_ids": gpu_ids,
        "device": args.device,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "max_rows": args.max_rows,
        "traditional_trials": args.traditional_trials,
        "dragonnet_trials": args.dragonnet_trials,
        "neural_max_epochs": args.neural_max_epochs,
        "seed": args.seed,
        "search_seed": args.search_seed,
        "objective": args.objective,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "cleaned_root": str(args.cleaned_root.expanduser().resolve()),
        "commands": {
            dataset: shlex.join(command)
            for dataset, command in commands.items()
        },
        "test_policy": (
            "This launcher performs validation tuning only. It never supplies "
            "--final-test."
        ),
    }
    _atomic_json(run_dir / "parallel_config.json", config)

    jobs = {
        dataset: {
            "state": "pending",
            "pid": None,
            "gpu_id": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "log": str(logs_dir / f"{dataset}.log"),
            "command": shlex.join(commands[dataset]),
        }
        for dataset in datasets
    }
    _write_status(
        status_path,
        started_at=started_at,
        run_dir=run_dir,
        args=args,
        datasets=datasets,
        exclusive=exclusive,
        jobs=jobs,
        state="dry_run" if args.dry_run else "running",
    )

    if args.dry_run:
        for dataset in datasets:
            print(f"[{dataset}] {jobs[dataset]['command']}")
        return 0

    pending = list(datasets)
    running: dict[int, dict] = {}
    free_gpus = list(gpu_ids)
    failures = 0

    while pending or running:
        launched = False
        while pending and len(running) < args.max_parallel:
            dataset = pending[0]
            running_datasets = {job["dataset"] for job in running.values()}
            if running_datasets & exclusive:
                break
            if dataset in exclusive and running:
                break
            if args.device != "cpu" and not free_gpus:
                break

            pending.pop(0)
            gpu_id = None
            if args.device != "cpu" and free_gpus:
                gpu_id = free_gpus.pop(0)

            log_path = logs_dir / f"{dataset}.log"
            log_stream = log_path.open("a", encoding="utf-8")
            log_stream.write(f"[{_now()}] {shlex.join(commands[dataset])}\n")
            log_stream.flush()
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            thread_count = str(args.cpu_threads_per_job)
            env["OMP_NUM_THREADS"] = thread_count
            env["MKL_NUM_THREADS"] = thread_count
            env["OPENBLAS_NUM_THREADS"] = thread_count

            try:
                process = subprocess.Popen(
                    commands[dataset],
                    cwd=Path(__file__).parent,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
            except Exception as exc:
                log_stream.write(f"[{_now()}] launch failed: {type(exc).__name__}: {exc}\n")
                log_stream.close()
                if gpu_id is not None:
                    free_gpus.append(gpu_id)
                jobs[dataset].update(
                    {
                        "state": "failed_to_start",
                        "finished_at": _now(),
                        "exit_code": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                failures += 1
                continue

            running[process.pid] = {
                "dataset": dataset,
                "process": process,
                "log_stream": log_stream,
                "gpu_id": gpu_id,
            }
            jobs[dataset].update(
                {
                    "state": "running",
                    "pid": process.pid,
                    "gpu_id": gpu_id,
                    "started_at": _now(),
                }
            )
            launched = True
            print(
                f"[{_now()}] START {dataset} pid={process.pid} "
                f"gpu={gpu_id if gpu_id is not None else 'none'}",
                flush=True,
            )
            _write_status(
                status_path,
                started_at=started_at,
                run_dir=run_dir,
                args=args,
                datasets=datasets,
                exclusive=exclusive,
                jobs=jobs,
                state="running",
            )

        finished_pids = []
        for pid, job in running.items():
            exit_code = job["process"].poll()
            if exit_code is None:
                continue
            finished_pids.append(pid)
            job["log_stream"].write(
                f"[{_now()}] process finished with exit code {exit_code}\n"
            )
            job["log_stream"].close()
            if job["gpu_id"] is not None:
                free_gpus.append(job["gpu_id"])
                free_gpus.sort(key=gpu_ids.index)
            dataset = job["dataset"]
            jobs[dataset].update(
                {
                    "state": "completed" if exit_code == 0 else "failed",
                    "finished_at": _now(),
                    "exit_code": exit_code,
                }
            )
            failures += int(exit_code != 0)
            print(
                f"[{_now()}] END {dataset} exit_code={exit_code}",
                flush=True,
            )

        for pid in finished_pids:
            del running[pid]

        _write_status(
            status_path,
            started_at=started_at,
            run_dir=run_dir,
            args=args,
            datasets=datasets,
            exclusive=exclusive,
            jobs=jobs,
            state="running" if pending or running else "completed",
        )
        if (pending or running) and not launched and not finished_pids:
            time.sleep(args.poll_seconds)

    final_state = "completed" if failures == 0 else "completed_with_failures"
    _write_status(
        status_path,
        started_at=started_at,
        run_dir=run_dir,
        args=args,
        datasets=datasets,
        exclusive=exclusive,
        jobs=jobs,
        state=final_state,
    )
    print(f"[{_now()}] {final_state}; run directory: {run_dir}", flush=True)
    return int(failures > 0)


def main() -> int:
    args = parse_args()
    datasets, exclusive, gpu_ids = _validate_args(args)
    run_dir = _new_run_dir(args)

    if args.detach:
        _detach(args, run_dir)
        return 0
    return _run_supervisor(args, run_dir, datasets, exclusive, gpu_ids)


if __name__ == "__main__":
    raise SystemExit(main())
