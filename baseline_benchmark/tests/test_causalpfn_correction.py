import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baseline_benchmark.models import make_model


def _binary_rct(seed=17, n=240, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    treatment = np.tile([0, 1, 0, 1], n // 4).astype(np.int8)
    probability = np.clip(
        0.25
        + 0.07 * X[:, 0]
        + treatment * (0.08 + 0.04 * X[:, 1]),
        0.02,
        0.98,
    )
    outcome = rng.binomial(1, probability).astype(np.int8)
    return X, treatment, outcome


class _FakeCATEEstimator:
    fit_sizes = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, *, X, t, y):
        self.X_train = np.asarray(X)
        self.t_train = np.asarray(t)
        self.y_train = np.asarray(y)
        self.fit_sizes.append(len(X))
        self.offset_ = float(y[t == 1].mean() - y[t == 0].mean())
        weights = np.linspace(0.01, 0.03, X.shape[1])
        self.weights_ = weights / max(np.linalg.norm(weights), 1e-12)
        self.icl_model = SimpleNamespace()
        self.stratifier = object()
        return self

    def estimate_cate(self, *, X):
        X = np.asarray(X)
        return self.offset_ + 0.03 * np.tanh(X @ self.weights_)


@pytest.mark.parametrize(
    "name",
    ["causalpfn_ridge_correction", "causalpfn_hgb_correction"],
)
def test_correction_models_use_oof_context_and_apply_residual(monkeypatch, name):
    _FakeCATEEstimator.fit_sizes = []
    monkeypatch.setitem(
        sys.modules,
        "causalpfn",
        SimpleNamespace(CATEEstimator=_FakeCATEEstimator),
    )
    X, treatment, outcome = _binary_rct()
    model = make_model(
        name,
        device="cpu",
        seed=5,
        correction_strength=0.5,
        correction_folds=2,
        correction_winsor_quantile=0.01,
        correction_max_iter=5,
        correction_min_samples_leaf=5,
        pseudo_max_iter=5,
        max_context_length=64,
        max_query_length=64,
        num_neighbours=16,
    )
    model.fit(X, treatment, outcome)
    components = model.predict_components(X[:21])

    assert _FakeCATEEstimator.fit_sizes == [120, 120, 240]
    assert components["cate"].shape == (21,)
    assert all(np.isfinite(values).all() for values in components.values())
    assert np.allclose(
        components["cate"],
        components["base_cate"] + 0.5 * components["correction"],
    )
    assert model.correction_diagnostics_.n_folds == 2
    assert model.correction_diagnostics_.n_correction_features == X.shape[1] + 3
    assert model.correction_diagnostics_.predicted_correction_std >= 0


def test_zero_strength_reproduces_base_causalpfn(monkeypatch):
    _FakeCATEEstimator.fit_sizes = []
    monkeypatch.setitem(
        sys.modules,
        "causalpfn",
        SimpleNamespace(CATEEstimator=_FakeCATEEstimator),
    )
    X, treatment, outcome = _binary_rct()
    model = make_model(
        "causalpfn_ridge_correction",
        device="cpu",
        seed=9,
        correction_strength=0.0,
        correction_folds=2,
        pseudo_max_iter=5,
    ).fit(X, treatment, outcome)
    components = model.predict_components(X[:17])

    assert np.array_equal(components["cate"], components["base_cate"])


def test_correction_parameter_validation():
    with pytest.raises(ValueError, match="correction_folds"):
        make_model("causalpfn_ridge_correction", correction_folds=1)
    with pytest.raises(ValueError, match="winsor"):
        make_model(
            "causalpfn_hgb_correction",
            correction_winsor_quantile=0.25,
        )
    with pytest.raises(ValueError, match="correction_strength"):
        make_model(
            "causalpfn_ridge_correction",
            correction_strength=-0.1,
        )
