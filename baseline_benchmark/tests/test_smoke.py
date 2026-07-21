import numpy as np
import pandas as pd

from baseline_benchmark.data import _make_preprocessor, _split_indices, upsample_training_data
from baseline_benchmark.metrics import evaluate_uplift
from baseline_benchmark.models import make_model


def synthetic_rct(seed=7, n=1200, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    t = rng.binomial(1, 0.5, size=n).astype(np.int8)
    tau = 0.18 * np.tanh(X[:, 0])
    p0 = 1 / (1 + np.exp(-(-2.0 + 0.4 * X[:, 1])))
    p = np.clip(p0 + t * tau, 0.001, 0.999)
    y = rng.binomial(1, p).astype(np.int8)
    return X, t, y, tau


def test_traditional_models_produce_finite_cate():
    X, t, y, _ = synthetic_rct()
    train, val, test = np.arange(0, 800), np.arange(800, 1000), np.arange(1000, 1200)
    for name in ("constant_ate", "s_learner", "t_learner", "x_learner", "dr_learner"):
        model = make_model(name, seed=3, max_iter=20, n_folds=3)
        model.fit(X[train], t[train], y[train], X[val], t[val], y[val])
        cate = model.predict_cate(X[test])
        assert cate.shape == (len(test),)
        assert np.isfinite(cate).all()


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


def test_constant_ranking_has_zero_incremental_curve_area():
    _, t, y, _ = synthetic_rct(n=4000)
    metrics = evaluate_uplift(y, np.zeros(len(y)), t)
    assert abs(metrics["qini_auc_normalized"]) < 1e-12
    assert abs(metrics["qini_coefficient"]) < 1e-12
    assert abs(metrics["uplift_auc_normalized"]) < 1e-12


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
    assert X_test.shape[1] == X_train.shape[1]
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()


def test_train_upsampling_balances_arms_without_changing_feature_width():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    t = np.array([0] * 4 + [1] * 8, dtype=np.int8)
    y = rng.binomial(1, 0.3, size=len(t)).astype(np.int8)
    X_fit, t_fit, y_fit, source = upsample_training_data(X, t, y, seed=5)
    assert X_fit.shape == (16, 3)
    assert y_fit.shape == t_fit.shape == (16,)
    assert np.sum(t_fit == 0) == np.sum(t_fit == 1) == 8
    assert np.isfinite(X_fit).all()
    assert np.isin(source, np.arange(len(X))).all()
