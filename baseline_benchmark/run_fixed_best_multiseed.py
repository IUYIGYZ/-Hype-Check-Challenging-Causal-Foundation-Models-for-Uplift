#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_benchmark.data import DATASET_SPECS, prepare_data
from baseline_benchmark.metrics import evaluate_uplift
from baseline_benchmark.models import available_models, make_model


MODEL_ORDER = ["t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn"]
OUTPUT_COLUMNS = [
    "dataset",
    "outcome",
    "model",
    "seed",
    "n_train",
    "n_validation",
    "n_test",
    "n_features",
    "fit_seconds",
    "predict_seconds",
    "qini_auc_normalized",
    "qini_coefficient",
    "uplift_auc_normalized",
    "auuc",
    "uplift_at_10pct",
    "uplift_at_20pct",
    "ate_test",
    "cate_mean",
    "cate_std",
    "test_events_control",
    "test_events_treated",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each dataset/model's frozen best validation parameters over "
            "multiple data/model seeds. This script never tunes hyperparameters."
        )
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Parallel tuning batch containing results/<dataset>/tuning_seed_*/.",
    )
    parser.add_argument("--datasets", default="hillstrom,lzd,retailhero,criteo")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "data_A_cleaned",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <batch-dir>/results/fixed_best_multiseed_seed0_4.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve frozen parameters and create placeholders without fitting models.",
    )
    return parser.parse_args()


def _parse_names(value: str, *, allowed: set[str], label: str) -> list[str]:
    names: list[str] = []
    for item in value.split(","):
        name = item.strip().lower()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError(f"At least one {label} is required")
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} values {unknown}; allowed={sorted(allowed)}")
    return names


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        seed = int(item.strip())
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _find_tuning_dir(batch_dir: Path, dataset: str) -> Path:
    candidates = sorted((batch_dir / "results" / dataset).glob("tuning_seed_*"))
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one tuning directory for {dataset}, found {candidates}"
        )
    return candidates[0]


def _load_frozen_specs(
    tuning_dir: Path, models: list[str]
) -> tuple[str, dict[str, dict[str, object] | None], dict[str, object]]:
    config_path = tuning_dir / "tuning_config.json"
    best_path = tuning_dir / "best_params.json"
    trials_path = tuning_dir / "trial_metrics.csv"
    for path in (config_path, best_path, trials_path):
        if not path.exists():
            raise RuntimeError(f"Required tuning artifact is missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    trials = pd.read_csv(trials_path)
    targets = config.get("trial_targets", {})
    outcome = str(config["outcome_resolved"])
    frozen: dict[str, dict[str, object] | None] = {}
    audit: dict[str, object] = {
        "tuning_dir": str(tuning_dir.resolve()),
        "outcome": outcome,
        "models": {},
    }

    for model in models:
        target = int(targets.get(model, 0))
        model_trials = trials.loc[trials["model"].astype(str) == model]
        attempted = {
            int(value)
            for value in model_trials["trial"].dropna().to_numpy()
        }
        successful = model_trials.loc[model_trials["status"] == "ok", "trial"].dropna()
        complete = target > 0 and set(range(target)).issubset(attempted)
        has_best = model in best and isinstance(best[model].get("params"), dict)
        available = bool(complete and has_best and len(successful) > 0)
        frozen[model] = dict(best[model]["params"]) if available else None
        audit["models"][model] = {
            "target_trials": target,
            "attempted_trials": len(attempted),
            "successful_trials": int(len(successful)),
            "complete": complete,
            "available_for_multiseed": available,
            "selected_trial": int(best[model]["trial"]) if available else None,
            "params": frozen[model],
        }
    return outcome, frozen, audit


def _empty_row(dataset: str, outcome: str, model: str, seed: int) -> dict[str, object]:
    row = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update({"dataset": dataset, "outcome": outcome, "model": model, "seed": seed})
    return row


def _row_complete(row: dict[str, object]) -> bool:
    return pd.notna(row.get("fit_seconds")) and pd.notna(
        row.get("qini_auc_normalized")
    )


def _sort_frame(
    rows: dict[tuple[str, int, str], dict[str, object]],
    datasets: list[str],
    seeds: list[int],
    models: list[str],
) -> pd.DataFrame:
    dataset_order = {name: index for index, name in enumerate(datasets)}
    seed_order = {seed: index for index, seed in enumerate(seeds)}
    model_order = {name: index for index, name in enumerate(models)}
    values = list(rows.values())
    values.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            seed_order[int(row["seed"])],
            model_order[str(row["model"])],
        )
    )
    return pd.DataFrame(values, columns=OUTPUT_COLUMNS)


def _write_checkpoint(
    csv_path: Path,
    rows: dict[tuple[str, int, str], dict[str, object]],
    datasets: list[str],
    seeds: list[int],
    models: list[str],
) -> None:
    frame = _sort_frame(rows, datasets, seeds, models)
    temporary = csv_path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)


def _load_or_initialize_rows(
    csv_path: Path,
    datasets: list[str],
    seeds: list[int],
    models: list[str],
    outcomes: dict[str, str],
) -> dict[tuple[str, int, str], dict[str, object]]:
    rows: dict[tuple[str, int, str], dict[str, object]] = {}
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        missing = sorted(set(OUTPUT_COLUMNS) - set(existing.columns))
        if missing:
            raise RuntimeError(f"Existing output is missing columns: {missing}")
        for row in existing[OUTPUT_COLUMNS].to_dict(orient="records"):
            key = (str(row["dataset"]), int(row["seed"]), str(row["model"]))
            rows[key] = row
    for dataset in datasets:
        for seed in seeds:
            for model in models:
                key = (dataset, seed, model)
                rows.setdefault(key, _empty_row(dataset, outcomes[dataset], model, seed))
    return rows


def _fill_data_fields(row: dict[str, object], data) -> None:
    row.update(
        {
            "outcome": data.outcome,
            "n_train": len(data.y_train),
            "n_validation": len(data.y_val),
            "n_test": len(data.y_test),
            "n_features": data.X_train.shape[1],
            "test_events_control": int(np.sum(data.y_test[data.t_test == 0])),
            "test_events_treated": int(np.sum(data.y_test[data.t_test == 1])),
        }
    )


def _cleanup_model(model_name: str, model=None) -> None:
    if model_name in {
        "dragonnet",
        "causalpfn",
        "causalpfn_head_ft",
        "causalpfn_hgb_correction",
        "causalpfn_ridge_correction",
        "causalpfn_x_learner",
    }:
        try:
            import torch

            if model is not None and hasattr(model, "model_"):
                model.model_.to("cpu")
            if model is not None and hasattr(model, "estimator_"):
                icl_model = getattr(model.estimator_, "icl_model", None)
                if icl_model is not None:
                    icl_model.to("cpu")
            effect_model = getattr(model, "effect_model_", None)
            if effect_model is not None and hasattr(effect_model, "estimator_"):
                icl_model = getattr(effect_model.estimator_, "icl_model", None)
                if icl_model is not None:
                    icl_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError):
            pass
    del model
    gc.collect()


def _evaluate_model(model_name: str, params: dict[str, object], data, seed: int, device: str):
    kwargs = dict(params)
    kwargs["seed"] = seed
    if model_name in {
        "dragonnet",
        "causalpfn",
        "causalpfn_head_ft",
        "causalpfn_hgb_correction",
        "causalpfn_ridge_correction",
        "causalpfn_x_learner",
    }:
        kwargs["device"] = device
    model = make_model(model_name, **kwargs)
    fit_start = time.perf_counter()
    model.fit(
        data.X_train,
        data.t_train,
        data.y_train,
        data.X_val,
        data.t_val,
        data.y_val,
    )
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    cate = np.asarray(model.predict_cate(data.X_test), dtype=float)
    predict_seconds = time.perf_counter() - predict_start
    if cate.shape != data.y_test.shape or not np.isfinite(cate).all():
        raise RuntimeError(f"{model_name} returned invalid CATE predictions")
    metrics = evaluate_uplift(data.y_test, cate, data.t_test)
    return model, metrics, fit_seconds, predict_seconds


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    datasets = _parse_names(
        args.datasets, allowed=set(DATASET_SPECS), label="dataset"
    )
    models = _parse_names(
        args.models, allowed=set(available_models()), label="model"
    )
    seeds = _parse_seeds(args.seeds)
    if args.max_rows < 0:
        raise ValueError("max_rows must be 0 or a positive integer")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")

    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        raise ValueError(f"Batch directory does not exist: {batch_dir}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else batch_dir / "results" / "fixed_best_multiseed_seed0_4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_models_metrics_seed0_4.csv"
    status_path = output_dir / "status.json"

    outcomes: dict[str, str] = {}
    frozen_by_dataset: dict[str, dict[str, dict[str, object] | None]] = {}
    parameter_audit: dict[str, object] = {}
    for dataset in datasets:
        tuning_dir = _find_tuning_dir(batch_dir, dataset)
        outcome, frozen, audit = _load_frozen_specs(tuning_dir, models)
        outcomes[dataset] = outcome
        frozen_by_dataset[dataset] = frozen
        parameter_audit[dataset] = audit

    _write_json(output_dir / "frozen_parameter_audit.json", parameter_audit)
    rows = _load_or_initialize_rows(csv_path, datasets, seeds, models, outcomes)
    _write_checkpoint(csv_path, rows, datasets, seeds, models)
    config = {
        "batch_dir": str(batch_dir),
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "cleaned_root": str(args.cleaned_root.resolve()),
        "all_cleaned_rows": args.max_rows == 0,
        "max_rows": None if args.max_rows == 0 else args.max_rows,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "device": args.device,
        "selection_source": "best validation parameters from seed-42 tuning",
        "evaluation_split": "test",
        "hyperparameter_search": False,
        "output_csv": str(csv_path),
    }
    _write_json(output_dir / "run_config.json", config)
    if args.dry_run:
        print(f"Dry run complete; placeholders written to {csv_path}", flush=True)
        return

    errors: list[dict[str, object]] = []
    max_rows = None if args.max_rows == 0 else args.max_rows
    total_available = sum(
        frozen_by_dataset[dataset][model] is not None
        for dataset in datasets
        for model in models
        for _ in seeds
    )
    completed_before = sum(_row_complete(row) for row in rows.values())
    _write_json(
        status_path,
        {
            "state": "running",
            "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(),
            "completed_before_resume": completed_before,
            "available_tasks": total_available,
            "output_csv": str(csv_path),
        },
    )

    try:
        for dataset in datasets:
            for seed in seeds:
                print(
                    f"PREPARE dataset={dataset} seed={seed} "
                    f"rows={'all' if max_rows is None else max_rows}",
                    flush=True,
                )
                data = prepare_data(
                    cleaned_root=args.cleaned_root,
                    dataset=dataset,
                    outcome=None,
                    max_rows=max_rows,
                    seed=seed,
                    val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction,
                )
                for model_name in models:
                    key = (dataset, seed, model_name)
                    row = rows[key]
                    _fill_data_fields(row, data)
                    params = frozen_by_dataset[dataset][model_name]
                    if params is None:
                        print(
                            f"PENDING dataset={dataset} seed={seed} model={model_name}: "
                            "tuning trials are incomplete; metrics remain blank",
                            flush=True,
                        )
                        _write_checkpoint(csv_path, rows, datasets, seeds, models)
                        continue
                    if _row_complete(row):
                        print(
                            f"SKIP completed dataset={dataset} seed={seed} model={model_name}",
                            flush=True,
                        )
                        continue
                    model = None
                    try:
                        print(
                            f"START dataset={dataset} seed={seed} model={model_name}",
                            flush=True,
                        )
                        model, metrics, fit_seconds, predict_seconds = _evaluate_model(
                            model_name, params, data, seed, args.device
                        )
                        row.update(
                            {
                                "fit_seconds": fit_seconds,
                                "predict_seconds": predict_seconds,
                                "ate_test": metrics["ate_observed"],
                                **{
                                    key: value
                                    for key, value in metrics.items()
                                    if key != "ate_observed"
                                },
                            }
                        )
                        _write_checkpoint(csv_path, rows, datasets, seeds, models)
                        print(
                            f"END dataset={dataset} seed={seed} model={model_name} "
                            f"qini={metrics['qini_auc_normalized']}",
                            flush=True,
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model": model_name,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        _write_json(output_dir / "errors.json", errors)
                        print(
                            f"ERROR dataset={dataset} seed={seed} model={model_name}: {exc}",
                            flush=True,
                        )
                    finally:
                        _cleanup_model(model_name, model)
                del data
                gc.collect()
    finally:
        _write_checkpoint(csv_path, rows, datasets, seeds, models)
        completed = sum(_row_complete(row) for row in rows.values())
        pending = len(rows) - completed
        _write_json(
            status_path,
            {
                "state": "completed_with_pending" if pending else "completed",
                "pid": os.getpid(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "completed_rows": completed,
                "pending_or_error_rows": pending,
                "errors": len(errors),
                "output_csv": str(csv_path),
            },
        )

    print(f"Fixed-parameter multi-seed results written to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
