from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .causalpfn import CausalPFNEstimator
from .models import (
    _classifier,
    _fit_classifier,
    _fit_regressor,
    _positive_probability,
    _regressor,
    _training_arrays,
)


@dataclass(frozen=True)
class DRPseudoLabelDiagnostics:
    n_folds: int
    propensity: float
    raw_y0_min: float
    raw_y0_max: float
    raw_y1_min: float
    raw_y1_max: float


def cross_fitted_dr_potential_outcomes(
    X,
    t,
    y,
    *,
    seed: int = 42,
    n_folds: int = 5,
    max_iter: int = 100,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    propensity_clip: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, DRPseudoLabelDiagnostics]:
    """Build leakage-safe, smoothed DR labels for both potential outcomes.

    Stage one obtains out-of-fold nuisance predictions and constructs the two
    AIPW signals. Stage two cross-fits regressors over those signals. The final
    labels are clipped to [0, 1], which is the support of this benchmark's
    binary outcomes and avoids unstable targets for CausalPFN's histogram loss.
    """

    X, t, y = _training_arrays(X, t, y)
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(
        n_folds, (int, np.integer)
    ) or n_folds < 2:
        raise ValueError("n_folds must be an integer greater than or equal to 2")
    propensity_clip = float(propensity_clip)
    if not np.isfinite(propensity_clip) or not 0 < propensity_clip < 0.5:
        raise ValueError("propensity_clip must be a finite number in (0, 0.5)")

    strata = np.char.add(t.astype(str), np.char.add("_", y.astype(str)))
    _, counts = np.unique(strata, return_counts=True)
    if int(counts.min()) < 2:
        raise ValueError(
            "Cross-fitted DR labels need at least two samples in every T x Y stratum"
        )
    folds = max(2, min(int(n_folds), int(counts.min())))
    propensity = float(
        np.clip(np.mean(t), propensity_clip, 1.0 - propensity_clip)
    )

    nuisance_splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    mu0_oof = np.zeros(len(y), dtype=float)
    mu1_oof = np.zeros(len(y), dtype=float)
    outcome_template = _classifier(
        seed, max_iter, max_leaf_nodes, learning_rate
    )
    from sklearn.base import clone

    for train_idx, hold_idx in nuisance_splitter.split(X, strata):
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

    raw_y0 = mu0_oof + (1 - t) * (y - mu0_oof) / (1 - propensity)
    raw_y1 = mu1_oof + t * (y - mu1_oof) / propensity

    # A second cross-fitting stage estimates E[phi_t | X] without letting a
    # row's noisy AIPW signal train its own pseudo-label.
    smoothing_splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed + 1
    )
    y0_hat = np.zeros(len(y), dtype=float)
    y1_hat = np.zeros(len(y), dtype=float)
    effect_template = _regressor(
        seed + 2, max_iter, max_leaf_nodes, learning_rate
    )
    for train_idx, hold_idx in smoothing_splitter.split(X, strata):
        reg0 = _fit_regressor(
            clone(effect_template), X[train_idx], raw_y0[train_idx]
        )
        reg1 = _fit_regressor(
            clone(effect_template), X[train_idx], raw_y1[train_idx]
        )
        y0_hat[hold_idx] = reg0.predict(X[hold_idx])
        y1_hat[hold_idx] = reg1.predict(X[hold_idx])

    y0_hat = np.clip(y0_hat, 0.0, 1.0).astype(np.float32)
    y1_hat = np.clip(y1_hat, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(y0_hat).all() or not np.isfinite(y1_hat).all():
        raise RuntimeError("Cross-fitted DR pseudo-labels contain NaN or Inf")

    diagnostics = DRPseudoLabelDiagnostics(
        n_folds=folds,
        propensity=propensity,
        raw_y0_min=float(raw_y0.min()),
        raw_y0_max=float(raw_y0.max()),
        raw_y1_min=float(raw_y1.min()),
        raw_y1_max=float(raw_y1.max()),
    )
    return y0_hat, y1_hat, diagnostics


def _positive_int(value, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class CausalPFNHeadFinetunedEstimator(CausalPFNEstimator):
    """Head-only domain adaptation with cross-fitted DR potential outcomes."""

    name = "causalpfn_head_ft"

    def __init__(
        self,
        *,
        finetune_epochs: int = 10,
        finetune_learning_rate: float = 1e-4,
        finetune_weight_decay: float = 1e-4,
        finetune_context_length: int = 1024,
        finetune_query_length: int = 256,
        finetune_tasks_per_epoch: int = 8,
        finetune_validation_tasks: int = 4,
        finetune_validation_fraction: float = 0.2,
        finetune_patience: int = 3,
        finetune_gradient_clip: float = 1.0,
        pseudo_folds: int = 5,
        pseudo_max_iter: int = 100,
        pseudo_max_leaf_nodes: int = 31,
        pseudo_learning_rate: float = 0.05,
        pseudo_propensity_clip: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.finetune_epochs = _positive_int(
            finetune_epochs, name="finetune_epochs"
        )
        self.finetune_context_length = _positive_int(
            finetune_context_length, name="finetune_context_length", minimum=4
        )
        self.finetune_query_length = _positive_int(
            finetune_query_length, name="finetune_query_length"
        )
        self.finetune_tasks_per_epoch = _positive_int(
            finetune_tasks_per_epoch, name="finetune_tasks_per_epoch"
        )
        self.finetune_validation_tasks = _positive_int(
            finetune_validation_tasks, name="finetune_validation_tasks"
        )
        self.finetune_patience = _positive_int(
            finetune_patience, name="finetune_patience"
        )
        self.pseudo_folds = _positive_int(
            pseudo_folds, name="pseudo_folds", minimum=2
        )
        self.pseudo_max_iter = _positive_int(
            pseudo_max_iter, name="pseudo_max_iter"
        )
        self.pseudo_max_leaf_nodes = _positive_int(
            pseudo_max_leaf_nodes, name="pseudo_max_leaf_nodes", minimum=2
        )

        self.finetune_learning_rate = float(finetune_learning_rate)
        self.finetune_weight_decay = float(finetune_weight_decay)
        self.finetune_validation_fraction = float(finetune_validation_fraction)
        self.finetune_gradient_clip = float(finetune_gradient_clip)
        self.pseudo_learning_rate = float(pseudo_learning_rate)
        self.pseudo_propensity_clip = float(pseudo_propensity_clip)
        for name, value, lower, inclusive in (
            ("finetune_learning_rate", self.finetune_learning_rate, 0.0, False),
            ("finetune_weight_decay", self.finetune_weight_decay, 0.0, True),
            ("finetune_gradient_clip", self.finetune_gradient_clip, 0.0, False),
            ("pseudo_learning_rate", self.pseudo_learning_rate, 0.0, False),
        ):
            valid = value >= lower if inclusive else value > lower
            if not np.isfinite(value) or not valid:
                operator = ">=" if inclusive else ">"
                raise ValueError(f"{name} must be finite and {operator} {lower}")
        if not 0 < self.finetune_validation_fraction < 0.5:
            raise ValueError("finetune_validation_fraction must be in (0, 0.5)")

    @staticmethod
    def _stratified_context(rng, indices, treatment, outcome, size):
        indices = np.asarray(indices)
        arm0 = indices[treatment[indices] == 0]
        arm1 = indices[treatment[indices] == 1]
        if len(arm0) == 0 or len(arm1) == 0:
            raise ValueError("Every fine-tuning context pool needs both treatment arms")
        size = min(int(size), len(indices))
        groups = [
            indices[(treatment[indices] == arm) & (outcome[indices] == event)]
            for arm in (0, 1)
            for event in (0, 1)
        ]
        nonempty = [group for group in groups if len(group)]
        if size < len(nonempty):
            raise ValueError(
                "finetune_context_length is too small to represent all T x Y strata"
            )

        # Force every available T x Y stratum into the context. This prevents the
        # official per-arm outcome standardization from collapsing on sparse
        # binary outcomes (notably Hillstrom).
        required = np.asarray(
            [rng.choice(group) for group in nonempty], dtype=int
        )
        remaining_size = size - len(required)
        if remaining_size:
            pool = np.setdiff1d(indices, required, assume_unique=False)
            additional = rng.choice(
                pool, size=remaining_size, replace=False
            )
            chosen = np.concatenate([required, additional])
        else:
            chosen = required
        rng.shuffle(chosen)
        return chosen

    def _task_tensors(self, torch, X, t, y, y0, y1, context_idx, query_idx):
        device = self.device_
        return (
            torch.as_tensor(X[context_idx], dtype=torch.float32, device=device)[
                None
            ],
            torch.as_tensor(t[context_idx], dtype=torch.float32, device=device)[
                None
            ],
            torch.as_tensor(y[context_idx], dtype=torch.float32, device=device)[
                None
            ],
            torch.as_tensor(X[query_idx], dtype=torch.float32, device=device)[None],
            torch.as_tensor(y0[query_idx], dtype=torch.float32, device=device)[None],
            torch.as_tensor(y1[query_idx], dtype=torch.float32, device=device)[None],
        )

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)
        y0_pseudo, y1_pseudo, diagnostics = cross_fitted_dr_potential_outcomes(
            X,
            t,
            y,
            seed=self.seed,
            n_folds=self.pseudo_folds,
            max_iter=self.pseudo_max_iter,
            max_leaf_nodes=self.pseudo_max_leaf_nodes,
            learning_rate=self.pseudo_learning_rate,
            propensity_clip=self.pseudo_propensity_clip,
        )
        self.pseudo_label_diagnostics_ = diagnostics

        # Loading and weak-neighbour fitting are intentionally identical to the
        # untouched zero-shot adapter.
        super().fit(X, t, y, X_val, t_val, y_val)

        import torch

        X_model = np.ascontiguousarray(self.estimator_.X_train, dtype=np.float32)
        t_model = np.ascontiguousarray(t, dtype=np.float32)
        y_model = np.ascontiguousarray(y, dtype=np.float32)
        strata = np.char.add(
            t.astype(str), np.char.add("_", y.astype(str))
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.finetune_validation_fraction,
            random_state=self.seed + 97,
        )
        meta_train, meta_validation = next(splitter.split(X, strata))
        if len(meta_train) < 4 or len(meta_validation) < 2:
            raise ValueError("Not enough rows for the inner fine-tuning split")

        icl_model = self.estimator_.icl_model
        for parameter in icl_model.parameters():
            parameter.requires_grad = False
        head = icl_model.model.head
        for parameter in head.parameters():
            parameter.requires_grad = True
        trainable = [p for p in head.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("CausalPFN prediction head has no trainable parameters")
        initial_head_state = {
            key: value.detach().cpu().clone()
            for key, value in head.state_dict().items()
        }

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.finetune_learning_rate,
            weight_decay=self.finetune_weight_decay,
        )
        rng = np.random.default_rng(self.seed + 211)
        query_size = min(
            self.finetune_query_length, max(1, len(meta_train) // 4)
        )
        context_size = min(
            self.finetune_context_length,
            self.max_context_length,
            len(meta_train) - query_size,
        )
        if context_size < 4:
            raise ValueError("Fine-tuning context must contain at least four rows")

        # Fixed inner-validation tasks make early stopping comparable by epoch.
        validation_tasks = []
        validation_query_size = min(
            self.finetune_query_length, len(meta_validation)
        )
        for _ in range(self.finetune_validation_tasks):
            context_idx = self._stratified_context(
                rng, meta_train, t, y, context_size
            )
            query_idx = rng.choice(
                meta_validation,
                size=validation_query_size,
                replace=False,
            )
            validation_tasks.append((context_idx, query_idx))

        best_loss = float("inf")
        best_state = None
        stale = 0
        history = []
        for epoch in range(self.finetune_epochs):
            icl_model.train()
            train_losses = []
            for _ in range(self.finetune_tasks_per_epoch):
                query_idx = rng.choice(
                    meta_train, size=query_size, replace=False
                )
                context_pool = np.setdiff1d(
                    meta_train, query_idx, assume_unique=False
                )
                context_idx = self._stratified_context(
                    rng, context_pool, t, y, context_size
                )
                tensors = self._task_tensors(
                    torch,
                    X_model,
                    t_model,
                    y_model,
                    y0_pseudo,
                    y1_pseudo,
                    context_idx,
                    query_idx,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = icl_model(*tensors).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("CausalPFN fine-tuning produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable, self.finetune_gradient_clip
                )
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            icl_model.eval()
            validation_losses = []
            cuda_devices = (
                [torch.cuda.current_device()] if self.device_ == "cuda" else []
            )
            # The official loss samples an intervention internally. Forking and
            # resetting RNG state makes validation comparable across epochs
            # without perturbing the training RNG stream.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.seed + 307)
                if self.device_ == "cuda":
                    torch.cuda.manual_seed_all(self.seed + 307)
                with torch.no_grad():
                    for context_idx, query_idx in validation_tasks:
                        tensors = self._task_tensors(
                            torch,
                            X_model,
                            t_model,
                            y_model,
                            y0_pseudo,
                            y1_pseudo,
                            context_idx,
                            query_idx,
                        )
                        validation_losses.append(
                            float(icl_model(*tensors).mean().cpu())
                        )
            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.finetune_patience:
                    break

        if best_state is None:
            raise RuntimeError("CausalPFN head fine-tuning produced no checkpoint")
        head.load_state_dict(best_state)
        icl_model.to(self.device_).eval()
        squared_delta = sum(
            float(
                torch.sum(
                    (value.detach().cpu() - initial_head_state[key]) ** 2
                )
            )
            for key, value in head.state_dict().items()
        )
        self.finetune_history_ = history
        self.best_finetune_validation_loss_ = best_loss
        self.finetune_epochs_run_ = len(history)
        self.head_parameter_delta_norm_ = float(np.sqrt(squared_delta))
        self.trainable_parameter_count_ = int(
            sum(parameter.numel() for parameter in trainable)
        )
        self.frozen_parameter_count_ = int(
            sum(
                parameter.numel()
                for parameter in icl_model.parameters()
                if not parameter.requires_grad
            )
        )
        return self
