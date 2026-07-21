import json

import numpy as np
import pandas as pd

from run_fixed_best_multiseed import (
    OUTPUT_COLUMNS,
    _empty_row,
    _load_frozen_specs,
    _parse_seeds,
    _sort_frame,
)


def _write_tuning_artifacts(path, attempted_dragonnet_trials):
    path.mkdir()
    (path / "tuning_config.json").write_text(
        json.dumps(
            {
                "outcome_resolved": "conversion",
                "trial_targets": {"t_learner": 2, "dragonnet": 2},
            }
        ),
        encoding="utf-8",
    )
    (path / "best_params.json").write_text(
        json.dumps(
            {
                "t_learner": {"trial": 1, "params": {"max_iter": 200}},
                "dragonnet": {"trial": 0, "params": {"epochs": 20}},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"model": "t_learner", "trial": 0, "status": "ok"},
        {"model": "t_learner", "trial": 1, "status": "ok"},
    ]
    rows.extend(
        {"model": "dragonnet", "trial": trial, "status": "ok"}
        for trial in attempted_dragonnet_trials
    )
    pd.DataFrame(rows).to_csv(path / "trial_metrics.csv", index=False)


def test_frozen_specs_require_all_trials_to_be_attempted(tmp_path):
    tuning_dir = tmp_path / "tuning"
    _write_tuning_artifacts(tuning_dir, attempted_dragonnet_trials=[0])
    outcome, specs, audit = _load_frozen_specs(
        tuning_dir, ["t_learner", "dragonnet"]
    )
    assert outcome == "conversion"
    assert specs["t_learner"] == {"max_iter": 200}
    assert specs["dragonnet"] is None
    assert audit["models"]["dragonnet"]["complete"] is False


def test_frozen_specs_accept_complete_trials(tmp_path):
    tuning_dir = tmp_path / "tuning"
    _write_tuning_artifacts(tuning_dir, attempted_dragonnet_trials=[0, 1])
    _, specs, audit = _load_frozen_specs(
        tuning_dir, ["t_learner", "dragonnet"]
    )
    assert specs["dragonnet"] == {"epochs": 20}
    assert audit["models"]["dragonnet"]["available_for_multiseed"] is True


def test_seed_parser_deduplicates_and_preserves_order():
    assert _parse_seeds("4,0,1,4") == [4, 0, 1]


def test_placeholder_and_output_column_order():
    rows = {
        ("demo", 0, "dragonnet"): _empty_row(
            "demo", "conversion", "dragonnet", 0
        )
    }
    frame = _sort_frame(rows, ["demo"], [0], ["dragonnet"])
    assert frame.columns.tolist() == OUTPUT_COLUMNS
    assert len(frame) == 1
    assert np.isnan(frame.loc[0, "qini_auc_normalized"])
