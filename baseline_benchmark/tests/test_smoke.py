import numpy as np
import pandas as pd
import pytest

from baseline_benchmark.data import _make_preprocessor, _split_indices, prepare_data
from baseline_benchmark.metrics import evaluate_uplift, uplift_at_k
from baseline_benchmark.models import available_models, make_model
from run_baselines import _evaluation_arrays


def synthetic_rct(seed=7, n=1200, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    t = rng.binomial(1, 0.5, size=n).astype(np.int8)
    tau = 0.18 * np.tanh(X[:, 0])
    p0 = 1 / (1 + np.exp(-(-2.0 + 0.4 * X[:, 1])))
    p = np.clip(p0 + t * tau, 0.001, 0.999)
    y = rng.binomial(1, p).astype(np.int8)
    return X, t, y, tau


def test_available_model_set_is_exactly_the_requested_four():
    assert available_models() == ["t_learner", "x_learner", "dr_learner", "dragonnet"]


def test_traditional_models_produce_finite_cate():
    X, t, y, _ = synthetic_rct()
    train, val, test = np.arange(0, 800), np.arange(800, 1000), np.arange(1000, 1200)
    for name in ("t_learner", "x_learner", "dr_learner"):
        model = make_model(name, seed=3, max_iter=20, n_folds=3)
        model.fit(X[train], t[train], y[train], X[val], t[val], y[val])
        cate = model.predict_cate(X[test])
        assert cate.shape == (len(test),)
        assert np.isfinite(cate).all()


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "dr_learner"])
def test_traditional_models_reject_a_missing_treatment_arm(name):
    X = np.ones((12, 3), dtype=np.float32)
    t = np.zeros(12, dtype=np.int8)
    y = np.tile([0, 1], 6).astype(np.int8)
    model = make_model(name, seed=3, max_iter=5, n_folds=2)
    with pytest.raises(ValueError, match="both 0 and 1"):
        model.fit(X, t, y)


def test_dr_learner_rejects_invalid_cross_fitting_parameters():
    with pytest.raises(ValueError, match="n_folds"):
        make_model("dr_learner", n_folds=1)
    with pytest.raises(ValueError, match="propensity_clip"):
        make_model("dr_learner", propensity_clip=0.5)


def test_metrics_accept_a_known_ranking():
    _, t, y, tau = synthetic_rct(n=4000)
    metrics = evaluate_uplift(y, tau, t)
    assert set(metrics) >= {
        "qini_auc_normalized",
        "qini_coefficient",
        "uplift_auc_normalized",
        "auuc",
        "uplift_at_10pct",
    }
    assert all(np.isfinite(v) for v in metrics.values())
    assert "ate_observed" in metrics
    assert "ate_test" not in metrics


def test_constant_ranking_has_zero_incremental_curve_area():
    _, t, y, _ = synthetic_rct(n=4000)
    metrics = evaluate_uplift(y, np.zeros(len(y)), t)
    assert abs(metrics["qini_auc_normalized"]) < 1e-12
    assert abs(metrics["qini_coefficient"]) < 1e-12
    assert abs(metrics["uplift_auc_normalized"]) < 1e-12


def test_metrics_reject_fractional_treatment_labels():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.4, 0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="binary treatment"):
        evaluate_uplift(y, score, np.array([0.5, 1.0, 0.0, 1.0]))


def test_uplift_at_k_uses_floor_for_fractional_cutoff():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    score = np.arange(11, 0, -1)
    assert uplift_at_k(y, score, treatment, 0.20) == 1.0


def test_group_safe_split_keeps_duplicate_feature_vectors_together():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(300, 3))
    X = pd.DataFrame(np.repeat(base, 2, axis=0), columns=["a", "b", "c"])
    t = np.tile([0, 1], len(base)).astype(np.int8)
    y = rng.binomial(1, 0.3, size=len(X)).astype(np.int8)
    train, val, test = _split_indices(X, t, y, True, 5, 0.2, 0.2)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    assert not np.intersect1d(groups[train], groups[val]).size
    assert not np.intersect1d(groups[train], groups[test]).size
    assert not np.intersect1d(groups[val], groups[test]).size


def test_categorical_encoder_is_fit_on_train_only_and_handles_unknown_values():
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0],
            "history_segment": ["low", "mid", "high", "low"],
            "zip_code": ["Urban", "Rural", "Urban", None],
        }
    )
    test = pd.DataFrame(
        {
            "numeric": [3.0],
            "history_segment": ["unseen"],
            "zip_code": ["Urban"],
        }
    )
    preprocessor = _make_preprocessor(train)
    X_train = preprocessor.fit_transform(train)
    X_test = preprocessor.transform(test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    assert X_test.shape[1] == X_train.shape[1]
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert "zip_code_None" not in feature_names


def test_prepare_data_runs_end_to_end(tmp_path):
    dataset_dir = tmp_path / "Hillstrom"
    dataset_dir.mkdir()
    n_rows = 240
    pd.DataFrame(
        {
            "epk_id": np.arange(n_rows),
            "T": np.tile([0, 0, 1, 1], n_rows // 4),
            "treatment_dt": pd.Series([pd.NaT] * n_rows, dtype="datetime64[ns]"),
            "recency": np.arange(n_rows) % 12,
            "history_segment": np.tile(["low", "mid", None, "high"], n_rows // 4),
        }
    ).to_parquet(dataset_dir / "features.parquet", index=False)
    pd.DataFrame(
        {
            "epk_id": np.arange(n_rows),
            "conversion": np.tile([0, 1, 0, 1], n_rows // 4),
        }
    ).to_parquet(dataset_dir / "outcomes.parquet", index=False)

    prepared = prepare_data(
        cleaned_root=tmp_path,
        dataset="hillstrom",
        max_rows=None,
        seed=7,
    )

    assert len(prepared.X_train) + len(prepared.X_val) + len(prepared.X_test) == n_rows
    assert prepared.X_train.shape[1] == prepared.X_val.shape[1] == prepared.X_test.shape[1]
    assert all(np.isfinite(matrix).all() for matrix in (prepared.X_train, prepared.X_val, prepared.X_test))


def test_evaluation_split_keeps_validation_and_test_separate():
    class Data:
        X_val = np.array([[1.0], [2.0]])
        id_val = np.array([10, 11])
        t_val = np.array([0, 1])
        y_val = np.array([0, 1])
        X_test = np.array([[3.0], [4.0], [5.0]])
        id_test = np.array([20, 21, 22])
        t_test = np.array([0, 1, 0])
        y_test = np.array([1, 0, 1])

    validation = _evaluation_arrays(Data(), "validation")
    test = _evaluation_arrays(Data(), "test")
    assert np.array_equal(validation[1], Data.id_val)
    assert np.array_equal(test[1], Data.id_test)
    assert len(validation[0]) == 2
    assert len(test[0]) == 3
