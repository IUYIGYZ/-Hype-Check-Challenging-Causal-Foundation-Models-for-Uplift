import json

from tune_baselines import _parse_models, _sample_params, _select_best


def test_requested_models_and_deduplication():
    assert _parse_models("t_learner,x_learner,t_learner,dr_learner,dragonnet") == [
        "t_learner",
        "x_learner",
        "dr_learner",
        "dragonnet",
    ]


def test_trial_zero_matches_current_defaults():
    assert _sample_params(
        "dr_learner", 0, search_seed=1, neural_max_epochs=99
    ) == {
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "learning_rate": 0.05,
        "n_folds": 5,
    }
    assert _sample_params(
        "dragonnet", 0, search_seed=1, neural_max_epochs=99
    )["epochs"] == 99


def test_sampling_is_deterministic_and_in_range():
    first = _sample_params(
        "dragonnet", 7, search_seed=2026, neural_max_epochs=30
    )
    second = _sample_params(
        "dragonnet", 7, search_seed=2026, neural_max_epochs=30
    )
    assert first == second
    assert first["epochs"] == 30
    assert first["batch_size"] in {256, 512, 1024}
    assert 3e-4 <= first["learning_rate"] <= 3e-3


def test_best_trial_uses_validation_objective_then_tie_breakers():
    common = {
        "dataset": "demo",
        "outcome": "Y",
        "model": "t_learner",
        "status": "ok",
        "objective": "qini_auc_normalized",
        "predict_seconds": 0.1,
    }
    rows = [
        {
            **common,
            "trial": 0,
            "params_json": json.dumps({"max_iter": 100}),
            "fit_seconds": 3.0,
            "qini_auc_normalized": 0.1,
            "uplift_at_10pct": 0.02,
        },
        {
            **common,
            "trial": 1,
            "params_json": json.dumps({"max_iter": 200}),
            "fit_seconds": 4.0,
            "qini_auc_normalized": 0.1,
            "uplift_at_10pct": 0.03,
        },
        {
            **common,
            "trial": 2,
            "status": "error",
            "params_json": json.dumps({"max_iter": 400}),
            "fit_seconds": 1.0,
            "qini_auc_normalized": 9.0,
            "uplift_at_10pct": 9.0,
        },
    ]
    best = _select_best(
        rows, objective="qini_auc_normalized", models=["t_learner"]
    )
    assert best["t_learner"]["trial"] == 1
    assert best["t_learner"]["params"] == {"max_iter": 200}
