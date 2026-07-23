from __future__ import annotations

import gc
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .causalpfn import CausalPFNEstimator
from .models import _feature_matrix, _training_arrays


@dataclass(frozen=True)
class CausalPFNXLearnerDiagnostics:
    n_folds: int
    propensity: float
    mu0_oof_min: float
    mu0_oof_max: float
    mu1_oof_min: float
    mu1_oof_max: float
    d0_min: float
    d0_max: float
    d0_std: float
    d1_min: float
    d1_max: float
    d1_std: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _positive_int(value, *, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class CausalPFNXLearner(CausalPFNEstimator):
    """X-Learner whose outcome and effect models are CausalPFN.

    CausalPFN is cross-fitted, so an observation is never part of the context
    used to construct its own imputed treatment effect. A second CausalPFN is
    fitted on D0 for controls and D1 for treated rows; its two potential-outcome
    predictions act as the standard tau0 and tau1 effect models.
    """

    name = "causalpfn_x_learner"

    def __init__(
        self,
        *,
        x_folds: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_folds = _positive_int(x_folds, name="x_folds", minimum=2)

    def _new_oof_causalpfn(self, fold: int) -> CausalPFNEstimator:
        return CausalPFNEstimator(
            seed=self.seed + 1000 + fold,
            device=self.device_name,
            model_path=self.model_path,
            cache_dir=self.cache_dir,
            max_context_length=self.max_context_length,
            max_query_length=self.max_query_length,
            num_neighbours=self.num_neighbours,
            calibrate=False,
            verbose=self.verbose,
        )

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        X = np.ascontiguousarray(X, dtype=np.float32)
        strata = np.char.add(t.astype(str), np.char.add("_", y.astype(str)))
        _, counts = np.unique(strata, return_counts=True)
        if int(counts.min()) < 2:
            raise ValueError(
                "CausalPFN X-Learner needs at least two samples in every "
                "T x Y stratum"
            )
        folds = min(self.x_folds, int(counts.min()))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 301,
        )
        mu0_oof = np.zeros(len(y), dtype=float)
        mu1_oof = np.zeros(len(y), dtype=float)
        for fold, (train_idx, hold_idx) in enumerate(splitter.split(X, strata)):
            fold_model = self._new_oof_causalpfn(fold)
            fold_model.fit(X[train_idx], t[train_idx], y[train_idx])
            mu0, mu1 = fold_model.predict_potential_outcomes(X[hold_idx])
            mu0_oof[hold_idx] = mu0
            mu1_oof[hold_idx] = mu1
            del fold_model
            gc.collect()

        control = t == 0
        treated = ~control
        d0 = mu1_oof[control] - y[control]
        d1 = y[treated] - mu0_oof[treated]
        d_observed = np.empty(len(y), dtype=np.float32)
        d_observed[control] = d0
        d_observed[treated] = d1
        self.effect_model_ = self._new_oof_causalpfn(10_000)
        self.effect_model_.fit_continuous_outcome(X, t, d_observed)
        self.propensity_ = float(np.clip(np.mean(t), 1e-3, 1.0 - 1e-3))
        self.n_features_in_ = X.shape[1]
        self.diagnostics_ = CausalPFNXLearnerDiagnostics(
            n_folds=folds,
            propensity=self.propensity_,
            mu0_oof_min=float(mu0_oof.min()),
            mu0_oof_max=float(mu0_oof.max()),
            mu1_oof_min=float(mu1_oof.min()),
            mu1_oof_max=float(mu1_oof.max()),
            d0_min=float(d0.min()),
            d0_max=float(d0.max()),
            d0_std=float(d0.std()),
            d1_min=float(d1.min()),
            d1_max=float(d1.max()),
            d1_std=float(d1.std()),
        )
        return self

    def predict_cate(self, X):
        if not hasattr(self, "propensity_"):
            raise RuntimeError(
                "causalpfn_x_learner must be fitted before predict_cate"
            )
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but causalpfn_x_learner was "
                f"fitted with {self.n_features_in_} features"
            )
        tau0, tau1 = self.effect_model_.predict_potential_outcomes(X)
        cate = self.propensity_ * tau0 + (1.0 - self.propensity_) * tau1
        if cate.shape != (len(X),) or not np.isfinite(cate).all():
            raise RuntimeError("causalpfn_x_learner returned invalid CATE predictions")
        return cate
