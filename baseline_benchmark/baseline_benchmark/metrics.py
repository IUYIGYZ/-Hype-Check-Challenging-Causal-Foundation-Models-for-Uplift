from __future__ import annotations

import math

import numpy as np


def _as_arrays(y_true, uplift, treatment):
    y_true = np.asarray(y_true).reshape(-1)
    uplift = np.asarray(uplift, dtype=float).reshape(-1)
    treatment = np.asarray(treatment).reshape(-1)
    if not (len(y_true) == len(uplift) == len(treatment)):
        raise ValueError("y_true, uplift, and treatment must have the same length")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Metrics currently require a binary outcome")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Metrics require both binary treatment arms")
    if not np.isfinite(uplift).all():
        raise ValueError("Scores contain NaN or Inf")
    return y_true.astype(float), uplift, treatment.astype(int)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """scikit-uplift's sklearn.metrics.auc equivalent."""
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y, x))


def uplift_curve(y_true, uplift, treatment):
    """Compute the scikit-uplift cumulative uplift curve."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()
    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]
    num_trmnt = np.cumsum(treatment)[threshold_indices]
    y_trmnt = np.cumsum(y_true_trmnt)[threshold_indices]
    num_all = threshold_indices + 1
    num_ctrl = num_all - num_trmnt
    y_ctrl = np.cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (
        np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0)
        - np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)
    ) * num_all
    if curve_values[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]
    return num_all.astype(float), curve_values.astype(float)


def perfect_uplift_curve(y_true, treatment):
    """Compute scikit-uplift's optimum uplift curve."""
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    _as_arrays(y_true, np.zeros(len(y_true)), treatment)
    control_responders = np.sum((y_true == 1) & (treatment == 0))
    treated_nonresponders = np.sum((y_true == 0) & (treatment == 1))
    summand = y_true if control_responders > treated_nonresponders else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand
    return uplift_curve(y_true, perfect_uplift, treatment)


def uplift_auc_score(y_true, uplift, treatment) -> float:
    """Compute normalized area under the scikit-uplift uplift curve."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    x_baseline = np.array([0.0, x_perfect[-1]])
    y_baseline = np.array([0.0, y_perfect[-1]])
    baseline_area = _auc(x_baseline, y_baseline)
    perfect_area = _auc(x_perfect, y_perfect) - baseline_area
    actual_area = _auc(x_actual, y_actual) - baseline_area
    return float(actual_area / perfect_area) if abs(perfect_area) > 1e-12 else float("nan")


def qini_curve(y_true, uplift, treatment):
    """Compute the scikit-uplift cumulative Qini curve."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()
    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0
    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]
    num_trmnt = np.cumsum(treatment)[threshold_indices]
    y_trmnt = np.cumsum(y_true_trmnt)[threshold_indices]
    num_all = threshold_indices + 1
    num_ctrl = num_all - num_trmnt
    y_ctrl = np.cumsum(y_true_ctrl)[threshold_indices]
    curve_values = y_trmnt - y_ctrl * np.divide(
        num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt, dtype=float), where=num_ctrl != 0
    )
    if curve_values[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]
    return num_all.astype(float), curve_values.astype(float)


def perfect_qini_curve(y_true, treatment, negative_effect=True):
    """Compute scikit-uplift's optimum Qini curve."""
    y_true, _, treatment = _as_arrays(y_true, np.zeros(len(y_true)), treatment)
    if not isinstance(negative_effect, bool):
        raise TypeError(f"negative_effect should be bool, got: {type(negative_effect)}")
    if negative_effect:
        perfect_score = y_true * treatment - y_true * (1 - treatment)
        return qini_curve(y_true, perfect_score, treatment)
    ratio_random = y_true[treatment == 1].sum() - (
        len(y_true[treatment == 1])
        * y_true[treatment == 0].sum()
        / len(y_true[treatment == 0])
    )
    return (
        np.array([0.0, ratio_random, len(y_true)]),
        np.array([0.0, ratio_random, ratio_random]),
    )


def qini_auc_score(y_true, uplift, treatment, negative_effect=True) -> float:
    """Compute normalized area under the scikit-uplift Qini curve."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    if not isinstance(negative_effect, bool):
        raise TypeError(f"negative_effect should be bool, got: {type(negative_effect)}")
    x_actual, y_actual = qini_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_qini_curve(y_true, treatment, negative_effect)
    x_baseline = np.array([0.0, x_perfect[-1]])
    y_baseline = np.array([0.0, y_perfect[-1]])
    baseline_area = _auc(x_baseline, y_baseline)
    perfect_area = _auc(x_perfect, y_perfect) - baseline_area
    actual_area = _auc(x_actual, y_actual) - baseline_area
    return float(actual_area / perfect_area) if abs(perfect_area) > 1e-12 else float("nan")


def uplift_at_k(y_true, uplift, treatment, strategy="overall", k=0.3):
    """Compute uplift at k using scikit-uplift's ``overall`` strategy."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    if strategy not in {"overall", "by_group"}:
        raise ValueError("strategy must be 'overall' or 'by_group'")
    n_samples = len(y_true)
    k_type = np.asarray(k).dtype.kind
    if k_type == "i":
        if k <= 0 or k >= n_samples:
            raise ValueError(f"k={k} must be in (0, {n_samples})")
    elif k_type == "f":
        if k <= 0 or k >= 1:
            raise ValueError(f"k={k} must be in (0, 1)")
    else:
        raise ValueError(f"Invalid value for k: {k}")

    order = np.argsort(uplift, kind="mergesort")[::-1]
    if strategy == "overall":
        n_size = int(n_samples * k) if k_type == "f" else int(k)
        selected = order[:n_size]
        score_ctrl = y_true[selected][treatment[selected] == 0].mean()
        score_trmnt = y_true[selected][treatment[selected] == 1].mean()
    else:
        n_ctrl = int((treatment == 0).sum() * k) if k_type == "f" else int(k)
        n_trmnt = int((treatment == 1).sum() * k) if k_type == "f" else int(k)
        if n_ctrl > (treatment == 0).sum() or n_trmnt > (treatment == 1).sum():
            raise ValueError("k exceeds the size of one treatment group")
        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()
    return float(score_trmnt - score_ctrl)


def qini_coefficient(y_true, uplift, treatment) -> float:
    """Unnormalized Qini area above the random-ranking line, scaled by N^2."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    x, curve = qini_curve(y_true, uplift, treatment)
    baseline = _auc(np.array([0.0, x[-1]]), np.array([0.0, curve[-1]]))
    return (_auc(x, curve) - baseline) / (len(y_true) ** 2)


def auuc(y_true, uplift, treatment) -> float:
    """Raw uplift-curve area scaled by N^2."""
    y_true, uplift, treatment = _as_arrays(y_true, uplift, treatment)
    x, curve = uplift_curve(y_true, uplift, treatment)
    return _auc(x, curve) / (len(y_true) ** 2)


def evaluate_uplift(y, score, treatment) -> dict[str, float]:
    y, score, treatment = _as_arrays(y, score, treatment)
    return {
        "qini_auc_normalized": qini_auc_score(y, score, treatment),
        "qini_coefficient": qini_coefficient(y, score, treatment),
        "uplift_auc_normalized": uplift_auc_score(y, score, treatment),
        "auuc": auuc(y, score, treatment),
        "uplift_at_10pct": uplift_at_k(y, score, treatment, strategy="overall", k=0.10),
        "uplift_at_20pct": uplift_at_k(y, score, treatment, strategy="overall", k=0.20),
        "ate_test": float(y[treatment == 1].mean() - y[treatment == 0].mean()),
        "cate_mean": float(np.mean(score)),
        "cate_std": float(np.std(score)),
    }
