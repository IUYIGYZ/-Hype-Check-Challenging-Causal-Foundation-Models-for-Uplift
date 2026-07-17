from __future__ import annotations

from copy import deepcopy
import os

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class _RepresentationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dragon: bool):
        super().__init__()
        self.dragon = dragon
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.propensity = nn.Linear(hidden_dim, 1) if dragon else None

    def forward(self, x):
        r = self.representation(x)
        y0, y1 = self.head0(r).squeeze(-1), self.head1(r).squeeze(-1)
        propensity = self.propensity(r).squeeze(-1) if self.propensity is not None else None
        return y0, y1, propensity


class _NeuralCATEEstimator:
    name = "neural"
    dragon = False

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
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device_name = device

    def _device(self):
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    @staticmethod
    def _loss(model, x, t, y, dragon: bool):
        y0, y1, propensity = model(x)
        observed = torch.where(t > 0.5, y1, y0)
        outcome_loss = nn.functional.binary_cross_entropy_with_logits(observed, y)
        if not dragon:
            return outcome_loss
        propensity_loss = nn.functional.binary_cross_entropy_with_logits(propensity, t)
        return outcome_loss + propensity_loss

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        if X_val is None or t_val is None or y_val is None:
            raise ValueError(f"{self.name} requires validation data for early stopping")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        device = self._device()
        self.model_ = _RepresentationNet(X.shape[1], self.hidden_dim, self.dragon).to(device)
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
                loss = self._loss(self.model_, xb, tb, yb, self.dragon)
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(self._loss(self.model_, xv, tv, yv, self.dragon).cpu())
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = deepcopy({k: v.detach().cpu() for k, v in self.model_.state_dict().items()})
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{self.name} did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.to(device).eval()
        self.device_ = device
        self.best_validation_loss_ = best_loss
        return self

    def predict_cate(self, X):
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


class TARNetEstimator(_NeuralCATEEstimator):
    name = "tarnet"
    dragon = False


class DragonNetEstimator(_NeuralCATEEstimator):
    name = "dragonnet"
    dragon = True
