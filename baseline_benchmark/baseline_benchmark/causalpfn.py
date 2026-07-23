from __future__ import annotations

import random

import numpy as np

from .models import _binary_vector, _feature_matrix, _training_arrays


class CausalPFNEstimator:
    """Adapter for the official ``causalpfn.CATEEstimator`` package.

    The adapter deliberately follows the same ``fit``/``predict_cate`` contract as
    the existing benchmark estimators. Validation data are accepted for interface
    compatibility but are not used: CausalPFN is a pretrained in-context model and
    has no task-specific hyperparameter fitting or early stopping.
    """

    name = "causalpfn"

    def __init__(
        self,
        seed: int = 42,
        device: str = "auto",
        model_path: str = "vdblm/causalpfn",
        cache_dir: str | None = None,
        max_context_length: int = 4096,
        max_query_length: int = 4096,
        num_neighbours: int = 1024,
        calibrate: bool = False,
        verbose: bool = False,
    ):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        for name, value in (
            ("max_context_length", max_context_length),
            ("max_query_length", max_query_length),
            ("num_neighbours", num_neighbours),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.seed = int(seed)
        self.device_name = device
        self.model_path = str(model_path)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.max_context_length = int(max_context_length)
        self.max_query_length = int(max_query_length)
        self.num_neighbours = int(num_neighbours)
        self.calibrate = bool(calibrate)
        self.verbose = bool(verbose)

    def _resolve_device(self, torch) -> str:
        if self.device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CausalPFN but is not available")
        return self.device_name

    def _fit_arrays(self, X, t, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        t = np.ascontiguousarray(t, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)

        try:
            import torch
            from causalpfn import CATEEstimator
        except (ImportError, OSError) as exc:
            raise ImportError(
                "CausalPFN is not installed. Install the benchmark requirements "
                "with `pip install -r requirements.txt`."
            ) from exc

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # The official implementation asks FAISS for this many neighbours in
        # each treatment arm. Clipping avoids invalid (-1) indices on small
        # smoke runs without changing its large-data behaviour.
        smallest_arm = int(min(np.sum(t == 0), np.sum(t == 1)))
        resolved_context = max(2, self.max_context_length)
        resolved_neighbours = min(
            self.num_neighbours,
            smallest_arm,
            max(1, resolved_context // 2),
        )

        kwargs = {
            "device": self._resolve_device(torch),
            "model_path": self.model_path,
            "max_context_length": resolved_context,
            "max_query_length": self.max_query_length,
            "num_neighbours": resolved_neighbours,
            "calibrate": self.calibrate,
            "verbose": self.verbose,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir

        self.estimator_ = CATEEstimator(**kwargs)
        self.estimator_.fit(X=X, t=t, y=y)
        self.n_features_in_ = X.shape[1]
        self.device_ = kwargs["device"]
        self.num_neighbours_ = resolved_neighbours
        return self

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        return self._fit_arrays(X, t, y)

    def fit_continuous_outcome(self, X, t, y):
        """Fit the official model with a finite continuous outcome.

        The public benchmark task has a binary response, while X-Learner's
        D0/D1 second-stage targets are continuous. The official CausalPFN model
        supports this case by standardizing outcomes internally.
        """
        X = _feature_matrix(X, name="X")
        t = _binary_vector(
            t,
            name="treatment",
            n_samples=len(X),
            require_both=True,
        )
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome must be a numeric vector") from exc
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and outcome must have the same number of samples")
        if not np.isfinite(y).all():
            raise ValueError("outcome contains NaN or Inf")
        return self._fit_arrays(X, t, y)

    def predict_cate(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("causalpfn must be fitted before predict_cate")
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but causalpfn was fitted with "
                f"{self.n_features_in_} features"
            )
        X = np.ascontiguousarray(X, dtype=np.float32)
        cate = np.asarray(self.estimator_.estimate_cate(X=X), dtype=float).reshape(-1)
        if cate.shape != (len(X),) or not np.isfinite(cate).all():
            raise RuntimeError("causalpfn returned invalid CATE predictions")
        return cate

    def predict_potential_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return CausalPFN estimates of E[Y(0)|X] and E[Y(1)|X].

        The upstream package currently exposes only their difference publicly.
        This adapter uses the same internal CEPO call as ``estimate_cate`` so
        X-learner-style imputation can retain both potential outcomes.
        """
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "causalpfn must be fitted before predict_potential_outcomes"
            )
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but causalpfn was fitted with "
                f"{self.n_features_in_} features"
            )
        X = np.ascontiguousarray(X, dtype=np.float32)
        estimator = self.estimator_
        X_query = X
        max_feature_size = getattr(estimator, "max_feature_size", None)
        if max_feature_size is not None and X_query.shape[1] > max_feature_size:
            X_query = estimator.x_dim_transformer.transform(X_query)
        n_samples = len(X_query)
        mu = np.asarray(
            estimator._predict_cepo(
                X_context=estimator.X_train,
                t_context=estimator.t_train,
                y_context=estimator.y_train,
                X_query=np.concatenate([X_query, X_query], axis=0),
                t_query=np.concatenate(
                    [
                        np.zeros(n_samples, dtype=np.float32),
                        np.ones(n_samples, dtype=np.float32),
                    ]
                ),
                temperature=estimator.prediction_temperature,
            ),
            dtype=float,
        ).reshape(-1)
        if mu.shape != (2 * n_samples,) or not np.isfinite(mu).all():
            raise RuntimeError("causalpfn returned invalid potential outcomes")
        return mu[:n_samples], mu[n_samples:]
