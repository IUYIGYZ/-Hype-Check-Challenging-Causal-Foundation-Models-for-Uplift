#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import dump

from baseline_benchmark.data import DATASET_SPECS, prepare_data
from baseline_benchmark.metrics import evaluate_uplift
from baseline_benchmark.models import available_models, make_model


DEFAULT_MODELS = "t_learner,x_learner,dr_learner,dragonnet,causalpfn"
TRADITIONAL_MODELS = {"t_learner", "x_learner", "dr_learner"}
MODEL_OFFSETS = {
    "t_learner": 10_000,
    "x_learner": 20_000,
    "dr_learner": 30_000,
    "dragonnet": 40_000,
    "causalpfn": 50_000,
}
OBJECTIVES = (
    "qini_auc_normalized",
    "qini_coefficient",
    "uplift_auc_normalized",
    "auuc",
    "uplift_at_10pct",
    "uplift_at_20pct",
)
METRIC_NAMES = (
    "qini_auc_normalized",
    "qini_coefficient",
    "uplift_auc_normalized",
    "auuc",
    "uplift_at_10pct",
    "uplift_at_20pct",
    "ate_observed",
    "cate_mean",
    "cate_std",
)
SEARCH_SPACE = {
    "t_learner": {
        "max_iter": [100, 150, 200, 300, 400],
        "max_leaf_nodes": [15, 31, 63],
        "learning_rate": "log-uniform[0.02, 0.15]",
    },
    "x_learner": {
        "max_iter": [100, 150, 200, 300, 400],
        "max_leaf_nodes": [15, 31, 63],
        "learning_rate": "log-uniform[0.02, 0.15]",
    },
    "dr_learner": {
        "max_iter": [100, 150, 200, 300, 400],
        "max_leaf_nodes": [15, 31, 63],
        "learning_rate": "log-uniform[0.02, 0.15]",
        "n_folds": [3, 5],
    },
    "dragonnet": {
        "epochs": "fixed by --neural-max-epochs; validation early stopping is active",
        "batch_size": [256, 512, 1024],
        "hidden_dim": [64, 128, 256],
        "learning_rate": "log-uniform[0.0003, 0.003]",
        "weight_decay": "log-uniform[0.000001, 0.001], plus 0",
        "patience": [10, 15, 20],
    },
    "causalpfn": {
        "parameters": "fixed pretrained estimator; no task-specific tuning",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Random-search tuning for T-Learner, X-Learner, DR-Learner, and "
            "DragonNet, plus fixed evaluation of pretrained CausalPFN. "
            "Validation is used for selection; test is untouched "
            "unless --final-test is explicitly supplied."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="retailhero")
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "data_A_cleaned",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent / "tuning_results",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Resume a previous tuning directory with the same data/search settings.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 uses every cleaned row (default); use a positive value only for a dry run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Data split and model seed.")
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=20, help="Trials per model.")
    parser.add_argument(
        "--traditional-trials",
        type=int,
        default=None,
        help="Optional override for each T/X/DR learner.",
    )
    parser.add_argument(
        "--dragonnet-trials",
        type=int,
        default=None,
        help="Optional override for DragonNet.",
    )
    parser.add_argument("--neural-max-epochs", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--objective", choices=OBJECTIVES, default="qini_auc_normalized")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="After tuning all models, refit each best setting and evaluate test exactly once.",
    )
    return parser.parse_args()


def _parse_models(value: str) -> list[str]:
    requested = []
    for item in value.split(","):
        name = item.strip().lower()
        if name and name not in requested:
            requested.append(name)
    if not requested:
        raise ValueError("At least one model must be requested")
    unknown = sorted(set(requested) - set(available_models()))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available={available_models()}")
    return requested


def _positive_count(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _target_trials(model_name: str, args) -> int:
    if model_name == "causalpfn":
        return 1
    if model_name == "dragonnet" and args.dragonnet_trials is not None:
        return args.dragonnet_trials
    if model_name in TRADITIONAL_MODELS and args.traditional_trials is not None:
        return args.traditional_trials
    return args.trials


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_params(
    model_name: str,
    trial: int,
    *,
    search_seed: int,
    neural_max_epochs: int,
) -> dict[str, object]:
    if trial < 0:
        raise ValueError("trial must be non-negative")
    if model_name not in MODEL_OFFSETS:
        raise ValueError(f"Unsupported model {model_name!r}")

    if trial == 0:
        if model_name == "causalpfn":
            return {}
        if model_name in {"t_learner", "x_learner"}:
            return {"max_iter": 150, "max_leaf_nodes": 31, "learning_rate": 0.05}
        if model_name == "dr_learner":
            return {
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "learning_rate": 0.05,
                "n_folds": 5,
            }
        return {
            "epochs": neural_max_epochs,
            "batch_size": 512,
            "hidden_dim": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 12,
        }

    if model_name == "causalpfn":
        raise ValueError("CausalPFN is pretrained and has exactly one fixed trial")

    rng = np.random.default_rng(search_seed + MODEL_OFFSETS[model_name] + trial * 1009)
    if model_name in TRADITIONAL_MODELS:
        params: dict[str, object] = {
            "max_iter": int(rng.choice([100, 150, 200, 300, 400])),
            "max_leaf_nodes": int(rng.choice([15, 31, 63])),
            "learning_rate": _log_uniform(rng, 0.02, 0.15),
        }
        if model_name == "dr_learner":
            params["n_folds"] = int(rng.choice([3, 5]))
        return params

    weight_decay = 0.0 if rng.random() < 0.2 else _log_uniform(rng, 1e-6, 1e-3)
    return {
        "epochs": neural_max_epochs,
        "batch_size": int(rng.choice([256, 512, 1024])),
        "hidden_dim": int(rng.choice([64, 128, 256])),
        "learning_rate": _log_uniform(rng, 3e-4, 3e-3),
        "weight_decay": weight_decay,
        "patience": int(rng.choice([10, 15, 20])),
    }


def _split_stats(t: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {
        "n": int(len(y)),
        "treatment_rate": float(np.mean(t)),
        "outcome_rate": float(np.mean(y)),
        "n_control": int(np.sum(t == 0)),
        "n_treated": int(np.sum(t == 1)),
        "events_control": int(np.sum(y[t == 0])),
        "events_treated": int(np.sum(y[t == 1])),
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_trials(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _select_best(
    trial_rows: list[dict[str, object]],
    *,
    objective: str,
    models: list[str],
) -> dict[str, dict[str, object]]:
    if not trial_rows:
        return {}
    frame = pd.DataFrame(trial_rows)
    best: dict[str, dict[str, object]] = {}
    for model_name in models:
        group = frame[(frame["model"] == model_name) & (frame["status"] == "ok")].copy()
        if group.empty:
            continue
        group[objective] = pd.to_numeric(group[objective], errors="coerce")
        group["uplift_at_10pct"] = pd.to_numeric(
            group["uplift_at_10pct"], errors="coerce"
        )
        group["fit_seconds"] = pd.to_numeric(group["fit_seconds"], errors="coerce")
        group = group[np.isfinite(group[objective])]
        if group.empty:
            continue
        group = group.sort_values(
            [objective, "uplift_at_10pct", "fit_seconds", "trial"],
            ascending=[False, False, True, True],
            na_position="last",
            kind="mergesort",
        )
        row = group.iloc[0]
        best[model_name] = {
            "trial": int(row["trial"]),
            "objective": objective,
            "objective_value": float(row[objective]),
            "params": json.loads(row["params_json"]),
            "validation_metrics": {
                metric: float(row[metric])
                for metric in METRIC_NAMES
                if metric in row and pd.notna(row[metric])
            },
            "fit_seconds": float(row["fit_seconds"]),
            "predict_seconds": float(row["predict_seconds"]),
        }
    return best


def _cleanup_model(model_name: str, model=None) -> None:
    if model_name in {"dragonnet", "causalpfn"}:
        try:
            import torch

            if model is not None and hasattr(model, "model_"):
                model.model_.to("cpu")
            if model is not None and hasattr(model, "estimator_"):
                icl_model = getattr(model.estimator_, "icl_model", None)
                if icl_model is not None:
                    icl_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError):
            pass
    del model
    gc.collect()


def _fit_and_evaluate(
    model_name: str,
    params: dict[str, object],
    data,
    *,
    evaluation_split: str,
    model_seed: int,
    device: str,
):
    kwargs = dict(params)
    kwargs["seed"] = model_seed
    if model_name in {"dragonnet", "causalpfn"}:
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

    if evaluation_split == "validation":
        X_eval, t_eval, y_eval, id_eval = (
            data.X_val,
            data.t_val,
            data.y_val,
            data.id_val,
        )
    elif evaluation_split == "test":
        X_eval, t_eval, y_eval, id_eval = (
            data.X_test,
            data.t_test,
            data.y_test,
            data.id_test,
        )
    else:
        raise ValueError(f"Unknown evaluation split {evaluation_split!r}")

    predict_start = time.perf_counter()
    cate = np.asarray(model.predict_cate(X_eval), dtype=float)
    predict_seconds = time.perf_counter() - predict_start
    if cate.shape != y_eval.shape or not np.isfinite(cate).all():
        raise RuntimeError(f"{model_name} returned invalid CATE predictions")
    metrics = evaluate_uplift(y_eval, cate, t_eval)
    return model, cate, id_eval, t_eval, y_eval, metrics, fit_seconds, predict_seconds


def _critical_resume_settings(args, models: list[str]) -> dict[str, object]:
    return {
        "dataset": args.dataset,
        "outcome_requested": args.outcome,
        "models": models,
        "cleaned_root": str(args.cleaned_root.resolve()),
        "max_rows_requested": args.max_rows,
        "seed": args.seed,
        "search_seed": args.search_seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "objective": args.objective,
        "device": args.device,
        "neural_max_epochs": args.neural_max_epochs,
    }


def _load_resume_rows(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "trial_metrics.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _prepare_output(args, models: list[str]) -> tuple[Path, dict[str, object] | None]:
    if args.resume_dir is not None:
        output_dir = args.resume_dir.resolve()
        if not output_dir.is_dir():
            raise ValueError(f"Resume directory does not exist: {output_dir}")
        config_path = output_dir / "tuning_config.json"
        if not config_path.exists():
            raise ValueError(f"Resume directory has no tuning_config.json: {output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        requested = _critical_resume_settings(args, models)
        previous_critical = previous.get("critical_settings")
        if previous_critical != requested:
            raise ValueError(
                "Resume settings differ from the original run. "
                f"original={previous_critical}, requested={requested}"
            )
        return output_dir, previous

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.dataset / f"tuning_seed_{args.seed}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, None


def main():
    args = parse_args()
    models = _parse_models(args.models)
    if args.max_rows < 0:
        raise ValueError("max_rows must be 0 or a positive integer")
    _positive_count(args.trials, name="trials")
    _positive_count(args.traditional_trials, name="traditional_trials")
    _positive_count(args.dragonnet_trials, name="dragonnet_trials")
    _positive_count(args.neural_max_epochs, name="neural_max_epochs")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be smaller than 1")

    output_dir, previous_config = _prepare_output(args, models)
    final_path = output_dir / "final_test_metrics.csv"
    if args.final_test and final_path.exists():
        raise RuntimeError(
            f"{final_path} already exists; refusing to evaluate test more than once"
        )
    max_rows = None if args.max_rows == 0 else args.max_rows
    scope = "ALL cleaned rows" if max_rows is None else f"at most {max_rows} rows"
    print(f"Preparing {scope}; test remains held out during tuning.", flush=True)
    data = prepare_data(
        cleaned_root=args.cleaned_root,
        dataset=args.dataset,
        outcome=args.outcome,
        max_rows=max_rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    data.split_table.to_parquet(output_dir / "splits.parquet", index=False)
    pd.DataFrame({"feature_name": data.feature_names}).to_csv(
        output_dir / "transformed_features.csv", index=False
    )
    dump(data.preprocessor, output_dir / "preprocessor.joblib")
    data_manifest = {
        "dataset": data.dataset,
        "outcome": data.outcome,
        "all_cleaned_rows": max_rows is None,
        "max_rows_resolved": max_rows,
        "n_transformed_features": len(data.feature_names),
        "group_safe_split": data.group_safe,
        "train": _split_stats(data.t_train, data.y_train),
        "validation": _split_stats(data.t_val, data.y_val),
        "test": _split_stats(data.t_test, data.y_test),
    }
    _write_json(output_dir / "data_manifest.json", data_manifest)

    trial_targets = {model: _target_trials(model, args) for model in models}
    config = {
        "critical_settings": _critical_resume_settings(args, models),
        "dataset_resolved": data.dataset,
        "outcome_resolved": data.outcome,
        "all_cleaned_rows": max_rows is None,
        "trial_targets": trial_targets,
        "search_space": {model: SEARCH_SPACE[model] for model in models},
        "selection_rule": (
            f"maximize validation {args.objective}; break ties with validation "
            "uplift_at_10pct, then lower fit time"
        ),
        "test_policy": (
            "test is never used by trials; --final-test evaluates each selected "
            "configuration once after all tuning"
        ),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "output_dir": str(output_dir),
        "resumed": previous_config is not None,
    }
    _write_json(output_dir / "tuning_config.json", config)

    trial_path = output_dir / "trial_metrics.csv"
    trial_rows = _load_resume_rows(output_dir)
    completed = {
        (str(row["model"]), int(row["trial"]))
        for row in trial_rows
        if "model" in row and "trial" in row and pd.notna(row["trial"])
    }

    for model_name in models:
        target = trial_targets[model_name]
        for trial in range(target):
            if (model_name, trial) in completed:
                print(f"SKIP completed {model_name} trial {trial}", flush=True)
                continue

            params = _sample_params(
                model_name,
                trial,
                search_seed=args.search_seed,
                neural_max_epochs=args.neural_max_epochs,
            )
            row: dict[str, object] = {
                "dataset": data.dataset,
                "outcome": data.outcome,
                "model": model_name,
                "trial": trial,
                "status": "error",
                "objective": args.objective,
                "params_json": json.dumps(params, sort_keys=True),
                "fit_seconds": float("nan"),
                "predict_seconds": float("nan"),
                "error": "",
            }
            model = None
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
                    model_name,
                    params,
                    data,
                    evaluation_split="validation",
                    model_seed=args.seed,
                    device=args.device,
                )
                objective_value = float(metrics[args.objective])
                if not np.isfinite(objective_value):
                    raise RuntimeError(
                        f"Validation objective {args.objective} is NaN or Inf"
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
                _cleanup_model(model_name, model)
                model = None

            trial_rows.append(row)
            _write_trials(trial_path, trial_rows)
            best = _select_best(trial_rows, objective=args.objective, models=models)
            _write_json(output_dir / "best_params.json", best)
            progress = {
                "model": model_name,
                "trial": trial,
                "status": row["status"],
                "objective": (
                    row.get(args.objective)
                    if row["status"] == "ok"
                    else None
                ),
                "params": params,
                "error": row["error"],
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    best = _select_best(trial_rows, objective=args.objective, models=models)
    _write_json(output_dir / "best_params.json", best)
    missing = [model for model in models if model not in best]
    if missing:
        raise RuntimeError(
            f"No successful finite-objective trial for models {missing}; "
            f"inspect {trial_path}"
        )

    print("Best validation configurations:", flush=True)
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)

    if args.final_test:
        if final_path.exists():
            raise RuntimeError(
                f"{final_path} already exists; refusing to evaluate test more than once"
            )
        prediction_dir = output_dir / "final_test_predictions"
        prediction_dir.mkdir(exist_ok=False)
        final_rows = []
        for model_name in models:
            params = best[model_name]["params"]
            model = None
            try:
                (
                    model,
                    cate,
                    ids,
                    treatment,
                    outcome,
                    metrics,
                    fit_seconds,
                    predict_seconds,
                ) = _fit_and_evaluate(
                    model_name,
                    params,
                    data,
                    evaluation_split="test",
                    model_seed=args.seed,
                    device=args.device,
                )
                final_row = {
                    "dataset": data.dataset,
                    "outcome": data.outcome,
                    "model": model_name,
                    "seed": args.seed,
                    "evaluation_split": "test",
                    "selected_validation_trial": best[model_name]["trial"],
                    "params_json": json.dumps(params, sort_keys=True),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    **metrics,
                }
                final_rows.append(final_row)
                pd.DataFrame(
                    {
                        "epk_id": ids,
                        "dataset": data.dataset,
                        "outcome": data.outcome,
                        "model": model_name,
                        "seed": args.seed,
                        "evaluation_split": "test",
                        "T": treatment,
                        "Y": outcome,
                        "cate_pred": cate,
                    }
                ).to_parquet(
                    prediction_dir / f"{model_name}.parquet",
                    index=False,
                )
                print(json.dumps(final_row, ensure_ascii=False), flush=True)
            finally:
                _cleanup_model(model_name, model)
                model = None
        pd.DataFrame(final_rows).to_csv(final_path, index=False)

    print(f"Tuning results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
