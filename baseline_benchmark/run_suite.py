#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "qini_auc_normalized",
    "qini_coefficient",
    "uplift_auc_normalized",
    "auuc",
    "uplift_at_10pct",
    "uplift_at_20pct",
    "fit_seconds",
    "predict_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a multi-dataset, multi-seed baseline suite.")
    parser.add_argument("--datasets", default="retailhero,lzd,hillstrom")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--models",
        default="t_learner,x_learner,dr_learner,dragonnet,causalpfn",
    )
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tree-max-iter", type=int, default=150)
    parser.add_argument("--tree-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--dr-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--causalpfn-model-path", default="vdblm/causalpfn")
    parser.add_argument("--causalpfn-cache-dir", type=Path, default=None)
    parser.add_argument("--causalpfn-max-context-length", type=int, default=4096)
    parser.add_argument("--causalpfn-max-query-length", type=int, default=4096)
    parser.add_argument("--causalpfn-num-neighbours", type=int, default=1024)
    parser.add_argument("--causalpfn-calibrate", action="store_true")
    parser.add_argument("--causalpfn-verbose", action="store_true")
    parser.add_argument("--causalpfn-ft-epochs", type=int, default=10)
    parser.add_argument("--causalpfn-ft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--causalpfn-ft-weight-decay", type=float, default=1e-4)
    parser.add_argument("--causalpfn-ft-context-length", type=int, default=1024)
    parser.add_argument("--causalpfn-ft-query-length", type=int, default=256)
    parser.add_argument("--causalpfn-ft-tasks-per-epoch", type=int, default=8)
    parser.add_argument("--causalpfn-ft-validation-tasks", type=int, default=4)
    parser.add_argument("--causalpfn-ft-validation-fraction", type=float, default=0.2)
    parser.add_argument("--causalpfn-ft-patience", type=int, default=3)
    parser.add_argument("--causalpfn-ft-gradient-clip", type=float, default=1.0)
    parser.add_argument("--causalpfn-pseudo-folds", type=int, default=5)
    parser.add_argument("--causalpfn-pseudo-max-iter", type=int, default=100)
    parser.add_argument("--causalpfn-pseudo-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--causalpfn-pseudo-learning-rate", type=float, default=0.05)
    parser.add_argument("--causalpfn-pseudo-propensity-clip", type=float, default=0.02)
    parser.add_argument("--causalpfn-correction-strength", type=float, default=0.5)
    parser.add_argument("--causalpfn-correction-folds", type=int, default=3)
    parser.add_argument("--causalpfn-correction-center", action="store_true")
    parser.add_argument(
        "--causalpfn-correction-winsor-quantile", type=float, default=0.01
    )
    parser.add_argument("--causalpfn-correction-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--causalpfn-correction-max-iter", type=int, default=50)
    parser.add_argument(
        "--causalpfn-correction-learning-rate", type=float, default=0.03
    )
    parser.add_argument(
        "--causalpfn-correction-max-leaf-nodes", type=int, default=15
    )
    parser.add_argument(
        "--causalpfn-correction-min-samples-leaf", type=int, default=200
    )
    parser.add_argument(
        "--causalpfn-correction-l2-regularization", type=float, default=1.0
    )
    parser.add_argument("--causalpfn-x-folds", type=int, default=3)
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="validation",
        help="Use validation for tuning; use test only after hyperparameters are frozen.",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "suite_results")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root / f"suite_{args.evaluation_split}_{stamp}"
    runs_dir = suite_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).parent / "run_baselines.py"

    for dataset in datasets:
        for seed in seeds:
            command = [
                sys.executable,
                str(runner),
                "--dataset",
                dataset,
                "--models",
                args.models,
                "--max-rows",
                str(args.max_rows),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--neural-learning-rate",
                str(args.neural_learning_rate),
                "--patience",
                str(args.patience),
                "--tree-max-iter",
                str(args.tree_max_iter),
                "--tree-max-leaf-nodes",
                str(args.tree_max_leaf_nodes),
                "--tree-learning-rate",
                str(args.tree_learning_rate),
                "--dr-folds",
                str(args.dr_folds),
                "--device",
                args.device,
                "--evaluation-split",
                args.evaluation_split,
                "--output-root",
                str(runs_dir),
            ]
            command.extend(
                [
                    "--causalpfn-model-path",
                    args.causalpfn_model_path,
                    "--causalpfn-max-context-length",
                    str(args.causalpfn_max_context_length),
                    "--causalpfn-max-query-length",
                    str(args.causalpfn_max_query_length),
                    "--causalpfn-num-neighbours",
                    str(args.causalpfn_num_neighbours),
                    "--causalpfn-ft-epochs",
                    str(args.causalpfn_ft_epochs),
                    "--causalpfn-ft-learning-rate",
                    str(args.causalpfn_ft_learning_rate),
                    "--causalpfn-ft-weight-decay",
                    str(args.causalpfn_ft_weight_decay),
                    "--causalpfn-ft-context-length",
                    str(args.causalpfn_ft_context_length),
                    "--causalpfn-ft-query-length",
                    str(args.causalpfn_ft_query_length),
                    "--causalpfn-ft-tasks-per-epoch",
                    str(args.causalpfn_ft_tasks_per_epoch),
                    "--causalpfn-ft-validation-tasks",
                    str(args.causalpfn_ft_validation_tasks),
                    "--causalpfn-ft-validation-fraction",
                    str(args.causalpfn_ft_validation_fraction),
                    "--causalpfn-ft-patience",
                    str(args.causalpfn_ft_patience),
                    "--causalpfn-ft-gradient-clip",
                    str(args.causalpfn_ft_gradient_clip),
                    "--causalpfn-pseudo-folds",
                    str(args.causalpfn_pseudo_folds),
                    "--causalpfn-pseudo-max-iter",
                    str(args.causalpfn_pseudo_max_iter),
                    "--causalpfn-pseudo-max-leaf-nodes",
                    str(args.causalpfn_pseudo_max_leaf_nodes),
                    "--causalpfn-pseudo-learning-rate",
                    str(args.causalpfn_pseudo_learning_rate),
                    "--causalpfn-pseudo-propensity-clip",
                    str(args.causalpfn_pseudo_propensity_clip),
                    "--causalpfn-correction-strength",
                    str(args.causalpfn_correction_strength),
                    "--causalpfn-correction-folds",
                    str(args.causalpfn_correction_folds),
                    "--causalpfn-correction-winsor-quantile",
                    str(args.causalpfn_correction_winsor_quantile),
                    "--causalpfn-correction-ridge-alpha",
                    str(args.causalpfn_correction_ridge_alpha),
                    "--causalpfn-correction-max-iter",
                    str(args.causalpfn_correction_max_iter),
                    "--causalpfn-correction-learning-rate",
                    str(args.causalpfn_correction_learning_rate),
                    "--causalpfn-correction-max-leaf-nodes",
                    str(args.causalpfn_correction_max_leaf_nodes),
                    "--causalpfn-correction-min-samples-leaf",
                    str(args.causalpfn_correction_min_samples_leaf),
                    "--causalpfn-correction-l2-regularization",
                    str(args.causalpfn_correction_l2_regularization),
                    "--causalpfn-x-folds",
                    str(args.causalpfn_x_folds),
                ]
            )
            if args.causalpfn_cache_dir is not None:
                command.extend(["--causalpfn-cache-dir", str(args.causalpfn_cache_dir)])
            if args.causalpfn_calibrate:
                command.append("--causalpfn-calibrate")
            if args.causalpfn_verbose:
                command.append("--causalpfn-verbose")
            if args.causalpfn_correction_center:
                command.append("--causalpfn-correction-center")
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)

    metric_files = sorted(runs_dir.glob("**/metrics.csv"))
    if not metric_files:
        raise RuntimeError("No metrics.csv files were produced")
    all_metrics = pd.concat((pd.read_csv(path) for path in metric_files), ignore_index=True)
    all_metrics.to_csv(suite_dir / "all_metrics.csv", index=False)

    rows = []
    group_columns = ["dataset", "outcome", "model", "evaluation_split"]
    for keys, group in all_metrics.groupby(group_columns, sort=True):
        row = {
            "dataset": keys[0],
            "outcome": keys[1],
            "model": keys[2],
            "evaluation_split": keys[3],
            "n_runs": len(group),
        }
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_halfwidth"] = ci95
        rows.append(row)
    pd.DataFrame(rows).to_csv(suite_dir / "summary.csv", index=False)
    (suite_dir / "suite_config.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "seeds": seeds,
                "models": args.models.split(","),
                "max_rows": args.max_rows,
                "epochs": args.epochs,
                "tree_max_iter": args.tree_max_iter,
                "tree_max_leaf_nodes": args.tree_max_leaf_nodes,
                "tree_learning_rate": args.tree_learning_rate,
                "dr_folds": args.dr_folds,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "neural_learning_rate": args.neural_learning_rate,
                "patience": args.patience,
                "device": args.device,
                "causalpfn_model_path": args.causalpfn_model_path,
                "causalpfn_cache_dir": (
                    str(args.causalpfn_cache_dir)
                    if args.causalpfn_cache_dir is not None
                    else None
                ),
                "causalpfn_max_context_length": args.causalpfn_max_context_length,
                "causalpfn_max_query_length": args.causalpfn_max_query_length,
                "causalpfn_num_neighbours": args.causalpfn_num_neighbours,
                "causalpfn_calibrate": args.causalpfn_calibrate,
                "causalpfn_head_finetune": {
                    "epochs": args.causalpfn_ft_epochs,
                    "learning_rate": args.causalpfn_ft_learning_rate,
                    "weight_decay": args.causalpfn_ft_weight_decay,
                    "context_length": args.causalpfn_ft_context_length,
                    "query_length": args.causalpfn_ft_query_length,
                    "tasks_per_epoch": args.causalpfn_ft_tasks_per_epoch,
                    "validation_tasks": args.causalpfn_ft_validation_tasks,
                    "validation_fraction": args.causalpfn_ft_validation_fraction,
                    "patience": args.causalpfn_ft_patience,
                    "gradient_clip": args.causalpfn_ft_gradient_clip,
                    "pseudo_folds": args.causalpfn_pseudo_folds,
                    "pseudo_max_iter": args.causalpfn_pseudo_max_iter,
                    "pseudo_max_leaf_nodes": args.causalpfn_pseudo_max_leaf_nodes,
                    "pseudo_learning_rate": args.causalpfn_pseudo_learning_rate,
                    "pseudo_propensity_clip": args.causalpfn_pseudo_propensity_clip,
                },
                "causalpfn_correction": {
                    "strength": args.causalpfn_correction_strength,
                    "folds": args.causalpfn_correction_folds,
                    "center": args.causalpfn_correction_center,
                    "winsor_quantile": (
                        args.causalpfn_correction_winsor_quantile
                    ),
                    "ridge_alpha": args.causalpfn_correction_ridge_alpha,
                    "max_iter": args.causalpfn_correction_max_iter,
                    "learning_rate": (
                        args.causalpfn_correction_learning_rate
                    ),
                    "max_leaf_nodes": (
                        args.causalpfn_correction_max_leaf_nodes
                    ),
                    "min_samples_leaf": (
                        args.causalpfn_correction_min_samples_leaf
                    ),
                    "l2_regularization": (
                        args.causalpfn_correction_l2_regularization
                    ),
                },
                "causalpfn_x_learner": {
                    "folds": args.causalpfn_x_folds,
                },
                "evaluation_split": args.evaluation_split,
                "python": sys.executable,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Suite summary written to {suite_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
