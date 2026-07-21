from pathlib import Path
from types import SimpleNamespace

import pytest

from run_parallel_tuning import _parse_csv, _tuning_command, _validate_args


def _args(**overrides):
    values = {
        "datasets": "retailhero,hillstrom",
        "exclusive_datasets": "criteo",
        "gpu_ids": "0,1",
        "max_parallel": 2,
        "max_rows": 0,
        "traditional_trials": 20,
        "dragonnet_trials": 20,
        "neural_max_epochs": 200,
        "cpu_threads_per_job": 8,
        "poll_seconds": 5.0,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "device": "cuda",
        "models": "t_learner,x_learner,dr_learner,dragonnet",
        "cleaned_root": Path("/tmp/cleaned"),
        "seed": 42,
        "search_seed": 2026,
        "objective": "qini_auc_normalized",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_csv_parser_deduplicates_and_normalizes():
    assert _parse_csv(" RetailHero,retailhero,HILLSTROM ") == [
        "retailhero",
        "hillstrom",
    ]


def test_default_criteo_exclusivity_is_ignored_for_a_subset():
    datasets, exclusive, gpu_ids = _validate_args(_args())
    assert datasets == ["retailhero", "hillstrom"]
    assert exclusive == set()
    assert gpu_ids == ["0", "1"]


def test_criteo_is_exclusive_when_it_is_requested():
    datasets, exclusive, _ = _validate_args(
        _args(datasets="retailhero,criteo")
    )
    assert datasets == ["retailhero", "criteo"]
    assert exclusive == {"criteo"}


def test_gpu_jobs_cannot_outnumber_assigned_gpus():
    with pytest.raises(ValueError, match="max_parallel"):
        _validate_args(_args(max_parallel=3, gpu_ids="0,1"))


def test_generated_command_uses_full_data_and_never_touches_test():
    args = _args()
    command = _tuning_command(args, "retailhero", Path("/tmp/results"))
    assert command[0]
    assert "--max-rows" in command
    assert command[command.index("--max-rows") + 1] == "0"
    assert "--final-test" not in command
    assert command[command.index("--dataset") + 1] == "retailhero"
    assert command[command.index("--device") + 1] == "cuda"
