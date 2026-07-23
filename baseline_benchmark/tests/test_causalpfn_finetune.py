import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baseline_benchmark.causalpfn_finetune import (
    cross_fitted_dr_potential_outcomes,
)
from baseline_benchmark.models import make_model


def _binary_rct(seed=11, n=240, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    treatment = np.tile([0, 1, 0, 1], n // 4).astype(np.int8)
    probability = np.clip(
        0.25 + 0.08 * X[:, 0] + treatment * (0.08 + 0.05 * X[:, 1]),
        0.02,
        0.98,
    )
    outcome = rng.binomial(1, probability).astype(np.int8)
    return X, treatment, outcome


def test_cross_fitted_dr_potential_outcomes_are_finite_and_deterministic():
    X, treatment, outcome = _binary_rct()
    first = cross_fitted_dr_potential_outcomes(
        X, treatment, outcome, seed=7, n_folds=3, max_iter=10
    )
    second = cross_fitted_dr_potential_outcomes(
        X, treatment, outcome, seed=7, n_folds=3, max_iter=10
    )
    y0, y1, diagnostics = first

    assert y0.shape == y1.shape == outcome.shape
    assert np.isfinite(y0).all() and np.isfinite(y1).all()
    assert np.all((0 <= y0) & (y0 <= 1))
    assert np.all((0 <= y1) & (y1 <= 1))
    assert np.array_equal(y0, second[0])
    assert np.array_equal(y1, second[1])
    assert diagnostics.n_folds == 3
    assert diagnostics.propensity == pytest.approx(0.5)


class _TinyICL:
    def __init__(self, torch, n_features):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(n_features, 6)
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(6, 6),
                    torch.nn.GELU(),
                    torch.nn.Linear(6, 1),
                )

            def forward(
                self,
                X_context,
                t_context,
                y_context,
                X_query,
                E_y0_query,
                E_y1_query,
            ):
                representation = self.backbone(X_query)
                prediction = self.head(representation).squeeze(-1)
                target = E_y1_query - E_y0_query
                return ((prediction - target) ** 2).mean(dim=-1)

        self.module = TinyModel()
        self.model = SimpleNamespace(
            head=self.module.head,
        )

    def parameters(self):
        return self.module.parameters()

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    def to(self, device):
        self.module.to(device)
        return self

    def __call__(self, *args):
        return self.module(*args)


class _TinyCATEEstimator:
    def __init__(self, **kwargs):
        self.device = kwargs["device"]

    def fit(self, *, X, t, y):
        import torch

        self.X_train = X
        self.t_train = t
        self.y_train = y
        self.icl_model = _TinyICL(torch, X.shape[1])
        self.stratifier = object()
        return self

    def estimate_cate(self, *, X):
        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32)
            representation = self.icl_model.module.backbone(tensor)
            return (
                self.icl_model.module.head(representation)
                .squeeze(-1)
                .cpu()
                .numpy()
            )


def test_head_only_finetune_updates_head_and_keeps_backbone_frozen(monkeypatch):
    import torch

    monkeypatch.setitem(
        sys.modules,
        "causalpfn",
        SimpleNamespace(CATEEstimator=_TinyCATEEstimator),
    )
    X, treatment, outcome = _binary_rct(n=120)
    model = make_model(
        "causalpfn_head_ft",
        device="cpu",
        seed=3,
        finetune_epochs=2,
        finetune_tasks_per_epoch=2,
        finetune_validation_tasks=1,
        finetune_context_length=16,
        finetune_query_length=8,
        finetune_patience=2,
        pseudo_folds=2,
        pseudo_max_iter=5,
    )
    model.fit(X, treatment, outcome)
    prediction = model.predict_cate(X[:9])

    assert prediction.shape == (9,)
    assert np.isfinite(prediction).all()
    assert model.trainable_parameter_count_ > 0
    assert model.frozen_parameter_count_ > 0
    assert model.head_parameter_delta_norm_ > 0
    assert all(
        not parameter.requires_grad
        for parameter in model.estimator_.icl_model.module.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.estimator_.icl_model.module.head.parameters()
    )
    assert 1 <= model.finetune_epochs_run_ <= 2
