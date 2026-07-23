import numpy as np
import pytest

from baseline_benchmark.causalpfn import CausalPFNEstimator
from baseline_benchmark.models import make_model


def _balanced_data(seed=11, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    t = np.tile([0, 0, 1, 1], n // 4).astype(np.int8)
    y = np.tile([0, 1, 0, 1], n // 4).astype(np.int8)
    return X, t, y


def test_causalpfn_xlearner_cross_fits_and_predicts(monkeypatch):
    context_sizes = []
    effect_context_sizes = []

    def fake_fit(self, X, t, y, *args):
        context_sizes.append(len(X))
        self.n_features_in_ = X.shape[1]
        self._context_shift = float(np.mean(y))
        self._is_effect_model = False
        return self

    def fake_continuous_fit(self, X, t, y):
        effect_context_sizes.append(len(X))
        assert np.any((y != 0) & (y != 1))
        self.n_features_in_ = X.shape[1]
        self._is_effect_model = True
        return self

    def fake_potential_outcomes(self, X):
        if self._is_effect_model:
            return 0.05 * X[:, 0], 0.08 * X[:, 0] + 0.02
        base = np.clip(0.25 + 0.05 * X[:, 1] + self._context_shift * 0.1, 0, 1)
        effect = 0.12 * np.tanh(X[:, 0])
        return base, np.clip(base + effect, 0, 1)

    monkeypatch.setattr(CausalPFNEstimator, "fit", fake_fit)
    monkeypatch.setattr(
        CausalPFNEstimator,
        "predict_potential_outcomes",
        fake_potential_outcomes,
    )
    monkeypatch.setattr(
        CausalPFNEstimator,
        "fit_continuous_outcome",
        fake_continuous_fit,
    )
    X, t, y = _balanced_data()
    model = make_model(
        "causalpfn_x_learner",
        seed=7,
        x_folds=4,
    )
    model.fit(X, t, y)
    cate = model.predict_cate(X[:13])

    assert context_sizes == [120, 120, 120, 120]
    assert effect_context_sizes == [160]
    assert cate.shape == (13,)
    assert np.isfinite(cate).all()
    assert model.diagnostics_.n_folds == 4
    assert model.diagnostics_.d0_std > 0
    assert model.diagnostics_.d1_std > 0
    assert hasattr(model, "effect_model_")


def test_causalpfn_xlearner_validates_cross_fitting_parameters():
    with pytest.raises(ValueError, match="x_folds"):
        make_model("causalpfn_x_learner", x_folds=1)
