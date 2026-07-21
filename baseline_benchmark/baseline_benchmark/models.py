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


class ConstantATE:
    name = "constant_ate"

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        self.ate_ = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
        return self

    def predict_cate(self, X):
        return np.full(len(X), self.ate_, dtype=float)


class SLearner:
    name = "s_learner"

    def __init__(self, seed=42, max_iter=150, max_leaf_nodes=31, learning_rate=0.05):
        self.model = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        self.model = _fit_classifier(self.model, np.column_stack([X, t]), y)
        return self

    def predict_cate(self, X):
        mu1 = _positive_probability(self.model, np.column_stack([X, np.ones(len(X))]))
        mu0 = _positive_probability(self.model, np.column_stack([X, np.zeros(len(X))]))
        return mu1 - mu0


class TLearner:
    name = "t_learner"

    def __init__(self, seed=42, max_iter=150, max_leaf_nodes=31, learning_rate=0.05):
        base = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        self.mu0, self.mu1 = clone(base), clone(base)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        self.mu0 = _fit_classifier(self.mu0, X[t == 0], y[t == 0])
        self.mu1 = _fit_classifier(self.mu1, X[t == 1], y[t == 1])
        return self

    def predict_cate(self, X):
        return _positive_probability(self.mu1, X) - _positive_probability(self.mu0, X)


class XLearner:
    name = "x_learner"

    def __init__(self, seed=42, max_iter=150, max_leaf_nodes=31, learning_rate=0.05):
        outcome = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        effect = _regressor(seed + 1, max_iter, max_leaf_nodes, learning_rate)
        self.mu0, self.mu1 = clone(outcome), clone(outcome)
        self.tau0, self.tau1 = clone(effect), clone(effect)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
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
        self.seed = seed
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.outcome_template = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        self.effect_model = _regressor(seed + 1, max_iter, max_leaf_nodes, learning_rate)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
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
        return np.asarray(self.effect_model.predict(X), dtype=float)


class CausalPFNEstimator:
    """Adapter for the pretrained CausalPFN CATE estimator.

    CausalPFN is a prior-fitted foundation model: ``fit`` only provides the
    current task as context and does not train a task-specific neural net.
    The optional dependency is imported lazily so the traditional baselines
    remain runnable without the larger CausalPFN installation.
    """

    name = "causalpfn"

    def __init__(self, device="auto", verbose=True):
        self.device_name = device
        self.verbose = verbose

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        try:
            import torch
            from causalpfn import CATEEstimator
        except ImportError as exc:
            raise ImportError(
                "The 'causalpfn' model requires the optional dependency. "
                "Install it with: pip install -r requirements-causalpfn.txt"
            ) from exc

        if self.device_name == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:0" if self.device_name == "cuda" else self.device_name)
        self.estimator_ = CATEEstimator(device=device, verbose=self.verbose)
        self.estimator_.fit(
            np.asarray(X, dtype=np.float32),
            np.asarray(t, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )
        self.device_ = str(device)
        return self

    def predict_cate(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("CausalPFN must be fitted before prediction")
        return np.asarray(
            self.estimator_.estimate_cate(np.asarray(X, dtype=np.float32)), dtype=float
        ).reshape(-1)


TRADITIONAL_MODELS = {
    "constant_ate": ConstantATE,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
}
NEURAL_MODELS = {"tarnet", "dragonnet"}
FOUNDATION_MODELS = {"causalpfn"}


def available_models() -> list[str]:
    return [*TRADITIONAL_MODELS, *sorted(NEURAL_MODELS), *sorted(FOUNDATION_MODELS)]


def make_model(name: str, **kwargs):
    key = name.lower()
    if key in TRADITIONAL_MODELS:
        cls = TRADITIONAL_MODELS[key]
        if cls is ConstantATE:
            return cls()
        allowed = {"seed", "max_iter", "max_leaf_nodes", "learning_rate"}
        if cls is DRLearner:
            allowed |= {"n_folds", "propensity_clip"}
        return cls(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in NEURAL_MODELS:
        from .neural import DragonNetEstimator, TARNetEstimator

        cls = TARNetEstimator if key == "tarnet" else DragonNetEstimator
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
        return cls(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in FOUNDATION_MODELS:
        return CausalPFNEstimator(
            device=kwargs.get("device", "auto"),
            verbose=kwargs.get("causalpfn_verbose", True),
        )
    raise ValueError(f"Unknown model {name!r}; choose from {available_models()}")
