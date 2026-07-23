import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baseline_benchmark.models import make_model


class _FakeCATEEstimator:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.icl_model = SimpleNamespace(to=lambda device: None)

    def fit(self, *, X, t, y):
        assert X.dtype == np.float32
        assert t.dtype == np.float32
        assert y.dtype == np.float32
        self.fitted = True
        return self

    def estimate_cate(self, *, X):
        assert self.fitted
        return X[:, 0] * 0.1


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None


def _install_fake_dependency(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=_FakeCuda(),
            manual_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "causalpfn",
        SimpleNamespace(CATEEstimator=_FakeCATEEstimator),
    )


def test_causalpfn_adapter_uses_official_api_and_clips_neighbours(monkeypatch):
    _install_fake_dependency(monkeypatch)
    X = np.arange(24, dtype=float).reshape(8, 3)
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    outcome = np.array([0, 1, 1, 0, 0, 1, 0, 1])

    model = make_model(
        "causalpfn",
        seed=7,
        device="auto",
        num_neighbours=1024,
        max_context_length=64,
    )
    model.fit(X, treatment, outcome)
    cate = model.predict_cate(X[:3])

    assert cate.shape == (3,)
    assert np.isfinite(cate).all()
    assert _FakeCATEEstimator.last_kwargs["device"] == "cpu"
    assert _FakeCATEEstimator.last_kwargs["num_neighbours"] == 4
    assert _FakeCATEEstimator.last_kwargs["max_context_length"] == 64


def test_causalpfn_adapter_validates_inputs_before_loading_dependency():
    model = make_model("causalpfn")
    with pytest.raises(ValueError, match="both 0 and 1"):
        model.fit(np.ones((8, 2)), np.zeros(8), np.zeros(8))


def test_causalpfn_adapter_rejects_wrong_feature_count(monkeypatch):
    _install_fake_dependency(monkeypatch)
    model = make_model("causalpfn", device="cpu")
    model.fit(
        np.ones((8, 3)),
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    with pytest.raises(ValueError, match="features"):
        model.predict_cate(np.ones((2, 2)))
