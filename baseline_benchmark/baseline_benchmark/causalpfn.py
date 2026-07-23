from __future__ import annotations

import random

import numpy as np

from .models import _feature_matrix, _training_arrays


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

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
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
