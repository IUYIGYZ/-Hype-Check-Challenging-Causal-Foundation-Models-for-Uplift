from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


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
    return values.astype(np.float32, copy=False)


def _positive_int(value, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


class _RepresentationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        head_dim = hidden_dim // 2
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ELU(), nn.Linear(head_dim, 1)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ELU(), nn.Linear(head_dim, 1)
        )
        self.propensity = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        representation = self.representation(x)
        y0 = self.head0(representation).squeeze(-1)
        y1 = self.head1(representation).squeeze(-1)
        propensity = self.propensity(representation).squeeze(-1)
        return y0, y1, propensity


class DragonNetEstimator:
    name = "dragonnet"

    def __init__(
        self,
        seed=42,
        epochs=100,
        batch_size=512,
        hidden_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=12,
        device="auto",
    ):
        self.seed = _positive_int(seed, name="seed", minimum=0)
        self.epochs = _positive_int(epochs, name="epochs")
        self.batch_size = _positive_int(batch_size, name="batch_size")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim", minimum=2)
        self.patience = _positive_int(patience, name="patience")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative finite number")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        self.device_name = device

    def _device(self):
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(self.device_name)

    @staticmethod
    def _loss(model, x, t, y):
        y0, y1, propensity = model(x)
        observed = torch.where(t > 0.5, y1, y0)
        outcome_loss = nn.functional.binary_cross_entropy_with_logits(observed, y)
        propensity_loss = nn.functional.binary_cross_entropy_with_logits(propensity, t)
        return outcome_loss + propensity_loss

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        if X_val is None or t_val is None or y_val is None:
            raise ValueError("dragonnet requires validation data for early stopping")

        X = _feature_matrix(X, name="X")
        t = _binary_vector(t, name="treatment", n_samples=len(X), require_both=True)
        y = _binary_vector(y, name="outcome", n_samples=len(X), require_both=False)
        X_val = _feature_matrix(X_val, name="X_val")
        if X_val.shape[1] != X.shape[1]:
            raise ValueError("X and X_val must have the same number of features")
        t_val = _binary_vector(
            t_val, name="validation treatment", n_samples=len(X_val), require_both=False
        )
        y_val = _binary_vector(
            y_val, name="validation outcome", n_samples=len(X_val), require_both=False
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._device()
        self.model_ = _RepresentationNet(X.shape[1], self.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        train_ds = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(t, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        tv = torch.as_tensor(t_val, dtype=torch.float32, device=device)
        yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)

        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model_.train()
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model_, xb, tb, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(self._loss(self.model_, xv, tv, yv).cpu())
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("dragonnet did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.to(device).eval()
        self.device_ = device
        self.n_features_in_ = X.shape[1]
        self.best_validation_loss_ = best_loss
        return self

    def predict_cate(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("dragonnet must be fitted before predict_cate")
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but dragonnet was fitted with "
                f"{self.n_features_in_} features"
            )

        scores = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size * 4):
                xb = torch.as_tensor(
                    X[start: start + self.batch_size * 4],
                    dtype=torch.float32,
                    device=self.device_,
                )
                y0, y1, _ = self.model_(xb)
                scores.append((torch.sigmoid(y1) - torch.sigmoid(y0)).cpu().numpy())
        return np.concatenate(scores).astype(float)
