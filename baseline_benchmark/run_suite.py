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
        default="constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet",
    )
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tree-max-iter", type=int, default=150)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "suite_results")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root / f"suite_{stamp}"
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
                "--tree-max-iter",
                str(args.tree_max_iter),
                "--device",
                args.device,
                "--output-root",
                str(runs_dir),
            ]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)

    metric_files = sorted(runs_dir.glob("**/metrics.csv"))
    if not metric_files:
        raise RuntimeError("No metrics.csv files were produced")
    all_metrics = pd.concat((pd.read_csv(path) for path in metric_files), ignore_index=True)
    all_metrics.to_csv(suite_dir / "all_metrics.csv", index=False)

    rows = []
    for keys, group in all_metrics.groupby(["dataset", "outcome", "model"], sort=True):
        row = {"dataset": keys[0], "outcome": keys[1], "model": keys[2], "n_runs": len(group)}
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
                "device": args.device,
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
