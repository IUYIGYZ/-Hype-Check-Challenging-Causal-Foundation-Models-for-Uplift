from __future__ import annotations

import math

import numpy as np


def _as_arrays(y, score, treatment):
    y = np.asarray(y, dtype=float).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    treatment = np.asarray(treatment, dtype=int).reshape(-1)
    if not (len(y) == len(score) == len(treatment)):
        raise ValueError("y, score, and treatment must have the same length")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Metrics currently require a binary outcome")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Metrics require both treatment arms")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain NaN or Inf")
    return y, score, treatment


def _threshold_indices(sorted_score: np.ndarray) -> np.ndarray:
    if len(sorted_score) == 1:
        return np.array([0], dtype=int)
    return np.r_[np.flatnonzero(np.diff(sorted_score)), len(sorted_score) - 1]


def qini_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Standard cumulative Qini gain curve with tied scores grouped together."""
    y, score, treatment = _as_arrays(y, score, treatment)
    order = np.argsort(-score, kind="mergesort")
    y, t, s = y[order], treatment[order], score[order]
    idx = _threshold_indices(s)
    n_t = np.cumsum(t)[idx].astype(float)
    n_c = np.cumsum(1 - t)[idx].astype(float)
    y_t = np.cumsum(y * t)[idx]
    y_c = np.cumsum(y * (1 - t))[idx]
    gain = np.zeros_like(y_t, dtype=float)
    valid = n_c > 0
    gain[valid] = y_t[valid] - y_c[valid] * n_t[valid] / n_c[valid]
    x = idx.astype(float) + 1.0
    return np.r_[0.0, x], np.r_[0.0, gain]


def _area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def qini_auc_normalized(y, score, treatment) -> float:
    """Normalized Qini AUC compatible with the scikit-uplift definition."""
    y, score, treatment = _as_arrays(y, score, treatment)
    x, actual = qini_curve(y, score, treatment)
    perfect_score = y * treatment - y * (1 - treatment)
    xp, perfect = qini_curve(y, perfect_score, treatment)
    baseline_actual = np.array([0.0, actual[-1]])
    baseline_perfect = np.array([0.0, perfect[-1]])
    actual_above = _area(x, actual) - _area(x[[0, -1]], baseline_actual)
    perfect_above = _area(xp, perfect) - _area(xp[[0, -1]], baseline_perfect)
    return float(actual_above / perfect_above) if abs(perfect_above) > 1e-12 else float("nan")


def qini_coefficient(y, score, treatment) -> float:
    """Unnormalized area above the random-ranking Qini line, scaled by N^2."""
    y, score, treatment = _as_arrays(y, score, treatment)
    x, gain = qini_curve(y, score, treatment)
    baseline = np.array([0.0, gain[-1]])
    return (_area(x, gain) - _area(x[[0, -1]], baseline)) / (len(y) ** 2)


def uplift_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative uplift-gain curve used by scikit-uplift."""
    y, score, treatment = _as_arrays(y, score, treatment)
    order = np.argsort(-score, kind="mergesort")
    y, t, s = y[order], treatment[order], score[order]
    idx = _threshold_indices(s)
    n_all = idx.astype(float) + 1.0
    n_t = np.cumsum(t)[idx].astype(float)
    n_c = n_all - n_t
    y_t = np.cumsum(y * t)[idx]
    y_c = np.cumsum(y * (1 - t))[idx]
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t != 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c != 0)
    gain = (rate_t - rate_c) * n_all
    return np.r_[0.0, n_all], np.r_[0.0, gain]


def _perfect_uplift_curve(y, treatment) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    control_responders = int(np.sum((y == 1) & (treatment == 0)))
    treated_nonresponders = int(np.sum((y == 0) & (treatment == 1)))
    summand = y if control_responders > treated_nonresponders else treatment
    perfect_score = 2 * (y == treatment) + summand
    return uplift_curve(y, perfect_score, treatment)


def uplift_auc_normalized(y, score, treatment) -> float:
    """Normalized uplift AUC compatible with the scikit-uplift definition."""
    y, score, treatment = _as_arrays(y, score, treatment)
    x, actual = uplift_curve(y, score, treatment)
    xp, perfect = _perfect_uplift_curve(y, treatment)
    baseline_x = np.array([0.0, xp[-1]])
    baseline_y = np.array([0.0, perfect[-1]])
    baseline_area = _area(baseline_x, baseline_y)
    actual_above = _area(x, actual) - baseline_area
    perfect_above = _area(xp, perfect) - baseline_area
    return float(actual_above / perfect_above) if abs(perfect_above) > 1e-12 else float("nan")


def uplift_at_k(y, score, treatment, fraction: float) -> float:
    y, score, treatment = _as_arrays(y, score, treatment)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(math.ceil(len(y) * fraction)))
    selected = np.argsort(-score, kind="mergesort")[:k]
    yt = y[selected][treatment[selected] == 1]
    yc = y[selected][treatment[selected] == 0]
    if not len(yt) or not len(yc):
        return float("nan")
    return float(yt.mean() - yc.mean())


def auuc(y, score, treatment) -> float:
    """Raw area under the standard uplift-gain curve, scaled by N^2."""
    y, score, treatment = _as_arrays(y, score, treatment)
    x, gain = uplift_curve(y, score, treatment)
    return _area(x, gain) / (len(y) ** 2)


def evaluate_uplift(y, score, treatment) -> dict[str, float]:
    y, score, treatment = _as_arrays(y, score, treatment)
    return {
        "qini_auc_normalized": qini_auc_normalized(y, score, treatment),
        "qini_coefficient": qini_coefficient(y, score, treatment),
        "uplift_auc_normalized": uplift_auc_normalized(y, score, treatment),
        "auuc": auuc(y, score, treatment),
        "uplift_at_10pct": uplift_at_k(y, score, treatment, 0.10),
        "uplift_at_20pct": uplift_at_k(y, score, treatment, 0.20),
        "ate_test": float(y[treatment == 1].mean() - y[treatment == 0].mean()),
        "cate_mean": float(np.mean(score)),
        "cate_std": float(np.std(score)),
    }
