from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold


def _classifier(seed: int, max_iter: int, max_leaf_nodes: int, learning_rate: float):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )


def _regressor(seed: int, max_iter: int, max_leaf_nodes: int, learning_rate: float):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )


def _fit_classifier(model, X: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
    return model.fit(X, y)


def _fit_regressor(model, X: np.ndarray, y: np.ndarray):
    if len(y) < 2 or float(np.std(y)) < 1e-12:
        model = DummyRegressor(strategy="mean")
    return model.fit(X, y)


def _positive_probability(model, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def _feature_matrix(X, *, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if len(X) == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"{name} must contain numeric features")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return X


def _binary_vector(values, *, name: str, n_samples: int, require_both: bool) -> np.ndarray:
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and encoded as 0/1") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if len(values) != n_samples:
        raise ValueError(f"X and {name} must have the same number of samples")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    unique = set(np.unique(values))
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name} must be binary and encoded as 0/1")
    if require_both and unique != {0.0, 1.0}:
        raise ValueError(f"{name} must contain both 0 and 1")
    return values.astype(np.int8, copy=False)


def _training_arrays(X, t, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _feature_matrix(X, name="X")
    t = _binary_vector(t, name="treatment", n_samples=len(X), require_both=True)
    y = _binary_vector(y, name="outcome", n_samples=len(X), require_both=False)
    return X, t, y


class TLearner:
    name = "t_learner"

    def __init__(self, seed=42, max_iter=150, max_leaf_nodes=31, learning_rate=0.05):
        base = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        self.mu0, self.mu1 = clone(base), clone(base)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        self.mu0 = _fit_classifier(self.mu0, X[t == 0], y[t == 0])
        self.mu1 = _fit_classifier(self.mu1, X[t == 1], y[t == 1])
        return self

    def predict_cate(self, X):
        X = _feature_matrix(X, name="X")
        return _positive_probability(self.mu1, X) - _positive_probability(self.mu0, X)


class XLearner:
    name = "x_learner"

    def __init__(self, seed=42, max_iter=150, max_leaf_nodes=31, learning_rate=0.05):
        outcome = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        effect = _regressor(seed + 1, max_iter, max_leaf_nodes, learning_rate)
        self.mu0, self.mu1 = clone(outcome), clone(outcome)
        self.tau0, self.tau1 = clone(effect), clone(effect)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        x0, x1 = X[t == 0], X[t == 1]
        y0, y1 = y[t == 0], y[t == 1]
        self.mu0 = _fit_classifier(self.mu0, x0, y0)
        self.mu1 = _fit_classifier(self.mu1, x1, y1)
        d0 = _positive_probability(self.mu1, x0) - y0
        d1 = y1 - _positive_probability(self.mu0, x1)
        self.tau0 = _fit_regressor(self.tau0, x0, d0)
        self.tau1 = _fit_regressor(self.tau1, x1, d1)
        self.propensity_ = float(np.clip(np.mean(t), 1e-3, 1 - 1e-3))
        return self

    def predict_cate(self, X):
        X = _feature_matrix(X, name="X")
        # Original X-learner weighting: g(x) * tau_0 + (1-g(x)) * tau_1.
        return self.propensity_ * self.tau0.predict(X) + (1 - self.propensity_) * self.tau1.predict(X)


class DRLearner:
    name = "dr_learner"

    def __init__(
        self,
        seed=42,
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.05,
        n_folds=5,
        propensity_clip=0.02,
    ):
        if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
            raise ValueError("n_folds must be an integer greater than or equal to 2")
        try:
            propensity_clip = float(propensity_clip)
        except (TypeError, ValueError) as exc:
            raise ValueError("propensity_clip must be a number in (0, 0.5)") from exc
        if not np.isfinite(propensity_clip) or not 0.0 < propensity_clip < 0.5:
            raise ValueError("propensity_clip must be a number in (0, 0.5)")
        self.seed = seed
        self.n_folds = int(n_folds)
        self.propensity_clip = propensity_clip
        self.outcome_template = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        self.effect_model = _regressor(seed + 1, max_iter, max_leaf_nodes, learning_rate)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        strata = np.char.add(t.astype(str), np.char.add("_", y.astype(str)))
        _, counts = np.unique(strata, return_counts=True)
        if int(counts.min()) < 2:
            raise ValueError("DR-Learner cross-fitting needs at least two samples in every T×Y stratum")
        folds = max(2, min(self.n_folds, int(counts.min())))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        mu0_oof = np.zeros(len(y), dtype=float)
        mu1_oof = np.zeros(len(y), dtype=float)
        propensity = float(np.clip(np.mean(t), self.propensity_clip, 1 - self.propensity_clip))
        for train_idx, hold_idx in splitter.split(X, strata):
            mu0 = _fit_classifier(
                clone(self.outcome_template),
                X[train_idx][t[train_idx] == 0],
                y[train_idx][t[train_idx] == 0],
            )
            mu1 = _fit_classifier(
                clone(self.outcome_template),
                X[train_idx][t[train_idx] == 1],
                y[train_idx][t[train_idx] == 1],
            )
            mu0_oof[hold_idx] = _positive_probability(mu0, X[hold_idx])
            mu1_oof[hold_idx] = _positive_probability(mu1, X[hold_idx])
        pseudo = (
            mu1_oof
            - mu0_oof
            + t * (y - mu1_oof) / propensity
            - (1 - t) * (y - mu0_oof) / (1 - propensity)
        )
        self.effect_model = _fit_regressor(self.effect_model, X, pseudo)
        return self

    def predict_cate(self, X):
        X = _feature_matrix(X, name="X")
        return np.asarray(self.effect_model.predict(X), dtype=float)


TRADITIONAL_MODELS = {
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
}
NEURAL_MODELS = {"dragonnet"}


def available_models() -> list[str]:
    return [*TRADITIONAL_MODELS, *sorted(NEURAL_MODELS)]


def make_model(name: str, **kwargs):
    key = name.lower()
    if key in TRADITIONAL_MODELS:
        cls = TRADITIONAL_MODELS[key]
        allowed = {"seed", "max_iter", "max_leaf_nodes", "learning_rate"}
        if cls is DRLearner:
            allowed |= {"n_folds", "propensity_clip"}
        return cls(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in NEURAL_MODELS:
        from .neural import DragonNetEstimator

        allowed = {
            "seed",
            "epochs",
            "batch_size",
            "hidden_dim",
            "learning_rate",
            "weight_decay",
            "patience",
            "device",
        }
        return DragonNetEstimator(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unknown model {name!r}; choose from {available_models()}")
