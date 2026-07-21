#!/usr/bin/env python
from __future__ import annotations

import argparse
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

from baseline_benchmark.data import DATASET_SPECS, prepare_data, upsample_training_data
from baseline_benchmark.metrics import evaluate_uplift
from baseline_benchmark.models import available_models, make_model


DEFAULT_MODELS = "constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet"


def parse_args():
    parser = argparse.ArgumentParser(description="Run fair uplift baseline comparisons.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="retailhero")
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model names")
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "data_A_cleaned",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--max-rows", type=int, default=50_000, help="0 means all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--tree-max-iter", type=int, default=150)
    parser.add_argument("--tree-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--dr-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--upsample-train",
        action="store_true",
        help="Balance treatment-arm sizes in training only by sampling with replacement.",
    )
    parser.add_argument(
        "--save-transformed-data",
        action="store_true",
        help="Also save numeric train/validation/test matrices as Parquet files.",
    )
    return parser.parse_args()


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


def _prepared_frame(X, ids, t, y, feature_names) -> pd.DataFrame:
    frame = pd.DataFrame(X, columns=feature_names)
    frame.insert(0, "Y", y)
    frame.insert(0, "T", t)
    frame.insert(0, "epk_id", ids)
    return frame


def main():
    args = parse_args()
    requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(requested) - set(available_models()))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available={available_models()}")
    max_rows = None if args.max_rows == 0 else args.max_rows
    data = prepare_data(
        cleaned_root=args.cleaned_root,
        dataset=args.dataset,
        outcome=args.outcome,
        max_rows=max_rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    X_train_fit, t_train_fit, y_train_fit = data.X_train, data.t_train, data.y_train
    train_source_rows = np.arange(len(data.y_train))
    if args.upsample_train:
        X_train_fit, t_train_fit, y_train_fit, train_source_rows = upsample_training_data(
            data.X_train, data.t_train, data.y_train, seed=args.seed
        )
    id_train_fit = data.id_train[train_source_rows]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.dataset / f"seed_{args.seed}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    data.split_table.to_parquet(output_dir / "splits.parquet", index=False)
    pd.DataFrame({"feature_name": data.feature_names}).to_csv(
        output_dir / "transformed_features.csv", index=False
    )
    dump(data.preprocessor, output_dir / "preprocessor.joblib")
    data_manifest = {
        "dataset": data.dataset,
        "outcome": data.outcome,
        "cleaned_root": str(args.cleaned_root.resolve()),
        "group_safe_split": data.group_safe,
        "seed": args.seed,
        "max_rows": max_rows,
        "n_transformed_features": len(data.feature_names),
        "transformed_feature_names": data.feature_names,
        "train": _split_stats(data.t_train, data.y_train),
        "train_fit": _split_stats(t_train_fit, y_train_fit),
        "upsample_train": bool(args.upsample_train),
        "validation": _split_stats(data.t_val, data.y_val),
        "test": _split_stats(data.t_test, data.y_test),
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.save_transformed_data:
        prepared_dir = output_dir / "prepared_data"
        prepared_dir.mkdir()
        for split_name, X, ids, t, y in (
            ("train", X_train_fit, id_train_fit, t_train_fit, y_train_fit),
            ("validation", data.X_val, data.id_val, data.t_val, data.y_val),
            ("test", data.X_test, data.id_test, data.t_test, data.y_test),
        ):
            _prepared_frame(X, ids, t, y, data.feature_names).to_parquet(
                prepared_dir / f"{split_name}.parquet", index=False
            )

    metrics_rows = []
    prediction_rows = []
    for model_name in requested:
        model = make_model(
            model_name,
            seed=args.seed,
            max_iter=args.tree_max_iter,
            max_leaf_nodes=args.tree_max_leaf_nodes,
            learning_rate=(
                args.neural_learning_rate
                if model_name in {"tarnet", "dragonnet"}
                else args.tree_learning_rate
            ),
            n_folds=args.dr_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            patience=args.patience,
            device=args.device,
            causalpfn_verbose=True,
        )
        fit_start = time.perf_counter()
        model.fit(
            X_train_fit,
            t_train_fit,
            y_train_fit,
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
        row = {
            "dataset": data.dataset,
            "outcome": data.outcome,
            "model": model_name,
            "seed": args.seed,
            "n_train": len(data.y_train),
            "n_train_fit": len(y_train_fit),
            "upsample_train": bool(args.upsample_train),
            "n_validation": len(data.y_val),
            "n_test": len(data.y_test),
            "n_features": data.X_train.shape[1],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            **evaluate_uplift(data.y_test, cate, data.t_test),
        }
        metrics_rows.append(row)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "epk_id": data.id_test,
                    "dataset": data.dataset,
                    "outcome": data.outcome,
                    "model": model_name,
                    "seed": args.seed,
                    "T": data.t_test,
                    "Y": data.y_test,
                    "cate_pred": cate,
                }
            )
        )
        print(json.dumps(row, ensure_ascii=False))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        output_dir / "predictions.parquet", index=False
    )
    config = vars(args).copy()
    config.update(
        {
            "models_resolved": requested,
            "outcome_resolved": data.outcome,
            "max_rows_resolved": max_rows,
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "output_dir": str(output_dir),
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
