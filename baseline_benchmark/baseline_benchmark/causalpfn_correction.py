from __future__ import annotations

import gc
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .causalpfn import CausalPFNEstimator
from .models import (
    _classifier,
    _feature_matrix,
    _fit_classifier,
    _positive_probability,
    _training_arrays,
)


@dataclass(frozen=True)
class CorrectionDiagnostics:
    correction_kind: str
    n_folds: int
    propensity: float
    n_correction_features: int
    dr_effect_min: float
    dr_effect_max: float
    residual_raw_min: float
    residual_raw_max: float
    residual_fit_min: float
    residual_fit_max: float
    residual_fit_std: float
    oof_causalpfn_std: float
    oof_dr_correlation: float
    predicted_correction_mean: float
    predicted_correction_std: float
    correction_strength: float
    correction_centered: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_int(value, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_float(value, *, name: str, lower: float, inclusive: bool) -> float:
    value = float(value)
    valid = value >= lower if inclusive else value > lower
    if not np.isfinite(value) or not valid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {operator} {lower}")
    return value


class BaseCausalPFNCorrection(CausalPFNEstimator):
    """Cross-fitted DR residual correction on top of zero-shot CausalPFN.

    Each outer-training row receives a CausalPFN prediction from a context that
    excludes its fold. The correction target is the cross-fitted AIPW/DR effect
    minus that OOF CausalPFN prediction. The final official CausalPFN estimator
    is then fitted on the complete outer-training split.
    """

    correction_kind: str

    def __init__(
        self,
        *,
        correction_strength: float = 0.5,
        correction_folds: int = 3,
        correction_center: bool = False,
        correction_winsor_quantile: float = 0.01,
        correction_ridge_alpha: float = 10.0,
        correction_max_iter: int = 50,
        correction_learning_rate: float = 0.03,
        correction_max_leaf_nodes: int = 15,
        correction_min_samples_leaf: int = 200,
        correction_l2_regularization: float = 1.0,
        pseudo_max_iter: int = 100,
        pseudo_max_leaf_nodes: int = 31,
        pseudo_learning_rate: float = 0.05,
        pseudo_propensity_clip: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.correction_strength = _finite_float(
            correction_strength,
            name="correction_strength",
            lower=0.0,
            inclusive=True,
        )
        self.correction_folds = _validated_int(
            correction_folds, name="correction_folds", minimum=2
        )
        self.correction_center = bool(correction_center)
        self.correction_winsor_quantile = float(correction_winsor_quantile)
        if (
            not np.isfinite(self.correction_winsor_quantile)
            or not 0.0 <= self.correction_winsor_quantile < 0.25
        ):
            raise ValueError(
                "correction_winsor_quantile must be finite and in [0, 0.25)"
            )
        self.correction_ridge_alpha = _finite_float(
            correction_ridge_alpha,
            name="correction_ridge_alpha",
            lower=0.0,
            inclusive=False,
        )
        self.correction_max_iter = _validated_int(
            correction_max_iter, name="correction_max_iter", minimum=1
        )
        self.correction_learning_rate = _finite_float(
            correction_learning_rate,
            name="correction_learning_rate",
            lower=0.0,
            inclusive=False,
        )
        self.correction_max_leaf_nodes = _validated_int(
            correction_max_leaf_nodes,
            name="correction_max_leaf_nodes",
            minimum=2,
        )
        self.correction_min_samples_leaf = _validated_int(
            correction_min_samples_leaf,
            name="correction_min_samples_leaf",
            minimum=2,
        )
        self.correction_l2_regularization = _finite_float(
            correction_l2_regularization,
            name="correction_l2_regularization",
            lower=0.0,
            inclusive=True,
        )
        self.pseudo_max_iter = _validated_int(
            pseudo_max_iter, name="pseudo_max_iter", minimum=1
        )
        self.pseudo_max_leaf_nodes = _validated_int(
            pseudo_max_leaf_nodes, name="pseudo_max_leaf_nodes", minimum=2
        )
        self.pseudo_learning_rate = _finite_float(
            pseudo_learning_rate,
            name="pseudo_learning_rate",
            lower=0.0,
            inclusive=False,
        )
        self.pseudo_propensity_clip = float(pseudo_propensity_clip)
        if (
            not np.isfinite(self.pseudo_propensity_clip)
            or not 0.0 < self.pseudo_propensity_clip < 0.5
        ):
            raise ValueError(
                "pseudo_propensity_clip must be finite and in (0, 0.5)"
            )

    def _new_oof_causalpfn(self, fold: int) -> CausalPFNEstimator:
        # Official temperature calibration changes confidence intervals, not
        # point CATE, so it is skipped for the expensive OOF contexts.
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

    def _make_correction_model(self, n_samples: int):
        if self.correction_kind == "ridge":
            return make_pipeline(
                StandardScaler(),
                Ridge(alpha=self.correction_ridge_alpha),
            )
        resolved_min_samples = min(
            self.correction_min_samples_leaf,
            max(2, n_samples // 10),
        )
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.correction_learning_rate,
            max_iter=self.correction_max_iter,
            max_leaf_nodes=self.correction_max_leaf_nodes,
            min_samples_leaf=resolved_min_samples,
            l2_regularization=self.correction_l2_regularization,
            early_stopping=False,
            random_state=self.seed + 2000,
        )

    @staticmethod
    def _correction_features(X, cate, mu0, mu1) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack([X, cate, mu0, mu1]),
            dtype=np.float32,
        )

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        X = np.ascontiguousarray(X, dtype=np.float32)
        strata = np.char.add(t.astype(str), np.char.add("_", y.astype(str)))
        _, counts = np.unique(strata, return_counts=True)
        if int(counts.min()) < 2:
            raise ValueError(
                "CausalPFN correction needs at least two samples in every "
                "T x Y stratum"
            )
        folds = max(2, min(self.correction_folds, int(counts.min())))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 301,
        )
        propensity = float(
            np.clip(
                np.mean(t),
                self.pseudo_propensity_clip,
                1.0 - self.pseudo_propensity_clip,
            )
        )
        outcome_template = _classifier(
            self.seed + 401,
            self.pseudo_max_iter,
            self.pseudo_max_leaf_nodes,
            self.pseudo_learning_rate,
        )
        mu0_oof = np.zeros(len(y), dtype=float)
        mu1_oof = np.zeros(len(y), dtype=float)
        cate_oof = np.zeros(len(y), dtype=float)

        for fold, (train_idx, hold_idx) in enumerate(
            splitter.split(X, strata)
        ):
            mu0 = _fit_classifier(
                clone(outcome_template),
                X[train_idx][t[train_idx] == 0],
                y[train_idx][t[train_idx] == 0],
            )
            mu1 = _fit_classifier(
                clone(outcome_template),
                X[train_idx][t[train_idx] == 1],
                y[train_idx][t[train_idx] == 1],
            )
            mu0_oof[hold_idx] = _positive_probability(mu0, X[hold_idx])
            mu1_oof[hold_idx] = _positive_probability(mu1, X[hold_idx])

            fold_model = self._new_oof_causalpfn(fold)
            fold_model.fit(X[train_idx], t[train_idx], y[train_idx])
            cate_oof[hold_idx] = fold_model.predict_cate(X[hold_idx])
            del fold_model
            gc.collect()

        dr_effect = (
            mu1_oof
            - mu0_oof
            + t * (y - mu1_oof) / propensity
            - (1 - t) * (y - mu0_oof) / (1 - propensity)
        )
        residual_raw = dr_effect - cate_oof
        quantile = self.correction_winsor_quantile
        if quantile > 0:
            lower, upper = np.quantile(
                residual_raw, [quantile, 1.0 - quantile]
            )
        else:
            lower, upper = float(residual_raw.min()), float(residual_raw.max())
        residual_fit = np.clip(residual_raw, lower, upper)

        correction_X = self._correction_features(
            X, cate_oof, mu0_oof, mu1_oof
        )
        self.correction_model_ = self._make_correction_model(len(X))
        self.correction_model_.fit(correction_X, residual_fit)
        train_correction = np.asarray(
            self.correction_model_.predict(correction_X), dtype=float
        )
        self.correction_center_ = (
            float(train_correction.mean()) if self.correction_center else 0.0
        )
        train_correction = np.clip(
            train_correction - self.correction_center_,
            lower,
            upper,
        )
        self.correction_lower_ = float(lower)
        self.correction_upper_ = float(upper)

        self.mu0_ = _fit_classifier(
            clone(outcome_template), X[t == 0], y[t == 0]
        )
        self.mu1_ = _fit_classifier(
            clone(outcome_template), X[t == 1], y[t == 1]
        )

        # The final point estimator uses the complete outer-training context.
        super().fit(X, t, y, X_val, t_val, y_val)
        correlation = (
            float(np.corrcoef(cate_oof, dr_effect)[0, 1])
            if np.std(cate_oof) > 0 and np.std(dr_effect) > 0
            else 0.0
        )
        self.correction_diagnostics_ = CorrectionDiagnostics(
            correction_kind=self.correction_kind,
            n_folds=folds,
            propensity=propensity,
            n_correction_features=correction_X.shape[1],
            dr_effect_min=float(dr_effect.min()),
            dr_effect_max=float(dr_effect.max()),
            residual_raw_min=float(residual_raw.min()),
            residual_raw_max=float(residual_raw.max()),
            residual_fit_min=float(residual_fit.min()),
            residual_fit_max=float(residual_fit.max()),
            residual_fit_std=float(residual_fit.std()),
            oof_causalpfn_std=float(cate_oof.std()),
            oof_dr_correlation=correlation,
            predicted_correction_mean=float(train_correction.mean()),
            predicted_correction_std=float(train_correction.std()),
            correction_strength=self.correction_strength,
            correction_centered=self.correction_center,
        )
        self.n_correction_features_in_ = correction_X.shape[1]
        return self

    def predict_components(self, X) -> dict[str, np.ndarray]:
        if not hasattr(self, "correction_model_"):
            raise RuntimeError(
                f"{self.name} must be fitted before predict_components"
            )
        X = _feature_matrix(X, name="X")
        base_cate = super().predict_cate(X)
        mu0 = _positive_probability(self.mu0_, X)
        mu1 = _positive_probability(self.mu1_, X)
        correction_X = self._correction_features(
            X, base_cate, mu0, mu1
        )
        correction = np.asarray(
            self.correction_model_.predict(correction_X), dtype=float
        )
        correction = np.clip(
            correction - self.correction_center_,
            self.correction_lower_,
            self.correction_upper_,
        )
        final_cate = base_cate + self.correction_strength * correction
        if not np.isfinite(final_cate).all():
            raise RuntimeError(f"{self.name} produced NaN or Inf predictions")
        return {
            "base_cate": base_cate,
            "correction": correction,
            "cate": final_cate,
        }

    def predict_cate(self, X):
        components = self.predict_components(X)
        self.last_prediction_components_ = components
        return components["cate"]


class CausalPFNRidgeCorrectionEstimator(BaseCausalPFNCorrection):
    name = "causalpfn_ridge_correction"
    correction_kind = "ridge"


class CausalPFNHGBCorrectionEstimator(BaseCausalPFNCorrection):
    name = "causalpfn_hgb_correction"
    correction_kind = "hgb"
