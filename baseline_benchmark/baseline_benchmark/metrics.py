from __future__ import annotations

import numpy as np
## metrics.py 不训练模型，也不调参。它接收每个模型预测的 CATE，然后判断：模型能不能把“真正更可能因为干预而购买的人”排到前面？
## CATE: Conditional Average Treatment Effect： 对某一类人/某一个人，干预平均能带来多少额外效果。

## input: y：实际结果，例如是否购买，取值为 0/1。treatment：是否接受干预，取值为 0/1。score：模型预测的 CATE/uplift。
def _as_arrays(y, score, treatment):   # 统一和检查输入
    y = np.asarray(y, dtype=float).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    treatment = np.asarray(treatment, dtype=float).reshape(-1)
    if not (len(y) == len(score) == len(treatment)):
        raise ValueError("y, score, and treatment must have the same length")
    if len(y) == 0:
        raise ValueError("y, score, and treatment must not be empty")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf")
    if not np.isfinite(treatment).all():
        raise ValueError("treatment contains NaN or Inf")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain NaN or Inf")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Metrics currently require a binary outcome")
    if not set(np.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("Metrics require a binary treatment encoded as 0/1")
    if set(np.unique(treatment)) != {0.0, 1.0}:
        raise ValueError("Metrics require both treatment arms")
    return y, score, treatment.astype(np.int8, copy=False)


def _threshold_indices(sorted_score: np.ndarray) -> np.ndarray:  # 并列分数处理,函数会把相同分数作为一个整体处理，只在分数变化的位置计算曲线：
# 这可以避免 Qini/Uplift 曲线因为相同分数用户的任意行顺序而产生不必要变化。
    if len(sorted_score) == 1:
        return np.array([0], dtype=int)
    return np.r_[np.flatnonzero(np.diff(sorted_score)), len(sorted_score) - 1]


def qini_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Standard cumulative Qini gain curve with tied scores grouped together."""
    y, score, treatment = _as_arrays(y, score, treatment)
    order = np.argsort(-score, kind="mergesort")   # 按照模型预测的 CATE 从大到小排序
    y, t, s = y[order], treatment[order], score[order]
    idx = _threshold_indices(s)
    n_t = np.cumsum(t)[idx].astype(float)  # treatment 用户数量
    n_c = np.cumsum(1 - t)[idx].astype(float)  # control 用户数量
    y_t = np.cumsum(y * t)[idx]  # treatment 中购买人数
    y_c = np.cumsum(y * (1 - t))[idx]  # control 中购买人数
    gain = np.zeros_like(y_t, dtype=float)
    valid = n_c > 0
    gain[valid] = y_t[valid] - y_c[valid] * n_t[valid] / n_c[valid]  # Qini Gain: 如果没有干预时的预期购买人数
    x = idx.astype(float) + 1.0
    return np.r_[0.0, x], np.r_[0.0, gain]


def _area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def qini_auc_normalized(y, score, treatment) -> float:  # 归一化 Qini AUC
    """Normalized Qini AUC compatible with the scikit-uplift definition."""
    ## 计算逻辑是: Normalized Qini = (AUC_{model}-AUC_{random})/ (AUC_{perfect}-AUC_{random})
    ## 解释：大于 0：比随机排序好；等于 0：与随机/常数排序相当；小于 0：排序方向可能有问题；越大通常越好。
    y, score, treatment = _as_arrays(y, score, treatment)
    x, actual = qini_curve(y, score, treatment)
    perfect_score = y * treatment - y * (1 - treatment)
    xp, perfect = qini_curve(y, perfect_score, treatment)
    baseline_actual = np.array([0.0, actual[-1]])
    baseline_perfect = np.array([0.0, perfect[-1]])
    actual_above = _area(x, actual) - _area(x[[0, -1]], baseline_actual)
    perfect_above = _area(xp, perfect) - _area(xp[[0, -1]], baseline_perfect)
    return float(actual_above / perfect_above) if abs(perfect_above) > 1e-12 else float("nan")



def qini_coefficient(y, score, treatment) -> float:  # Qini 系数
    """Unnormalized area above the random-ranking Qini line, scaled by N^2."""
    ## 作用： 衡量模型排序的好坏，越大通常越好。
    y, score, treatment = _as_arrays(y, score, treatment)
    x, gain = qini_curve(y, score, treatment)
    baseline = np.array([0.0, gain[-1]])
    return (_area(x, gain) - _area(x[[0, -1]], baseline)) / (len(y) ** 2)


def uplift_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative uplift-gain curve used by scikit-uplift."""
    ## treatment组购买率 - control组购买率，作用：衡量模型排序的好坏，越大通常越好。
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
    ## 越大越好；0 附近表示接近随机排序；负值表示排序较差
    y, score, treatment = _as_arrays(y, score, treatment)
    x, actual = uplift_curve(y, score, treatment)
    xp, perfect = _perfect_uplift_curve(y, treatment)
    baseline_x = np.array([0.0, xp[-1]])
    baseline_y = np.array([0.0, perfect[-1]])
    baseline_area = _area(baseline_x, baseline_y)
    actual_above = _area(x, actual) - baseline_area
    perfect_above = _area(xp, perfect) - baseline_area
    return float(actual_above / perfect_above) if abs(perfect_above) > 1e-12 else float("nan")


def uplift_at_k(y, score, treatment, fraction: float) -> float:   # 它只选择预测 CATE 最大的前一部分用户
    y, score, treatment = _as_arrays(y, score, treatment)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    # Match scikit-uplift's overall strategy: fractional k is floored.
    # Keep one row for very small evaluation samples.
    k = max(1, int(len(y) * fraction))
    selected = np.argsort(-score, kind="mergesort")[:k]
    yt = y[selected][treatment[selected] == 1]
    yc = y[selected][treatment[selected] == 0]
    if not len(yt) or not len(yc):
        return float("nan")
    return float(yt.mean() - yc.mean())


def auuc(y, score, treatment) -> float:
    """Raw area under the standard uplift-gain curve, scaled by N^2."""
    ## 计算整条 Uplift Gain Curve 的面积
    y, score, treatment = _as_arrays(y, score, treatment)
    x, gain = uplift_curve(y, score, treatment)
    return _area(x, gain) / (len(y) ** 2)


def evaluate_uplift(y, score, treatment) -> dict[str, float]:
    y, score, treatment = _as_arrays(y, score, treatment)
    return {
        "qini_auc_normalized": qini_auc_normalized(y, score, treatment),  # 作为主要调参指标
        "qini_coefficient": qini_coefficient(y, score, treatment),
        "uplift_auc_normalized": uplift_auc_normalized(y, score, treatment),  # 检查结论是否一致
        "auuc": auuc(y, score, treatment),
        "uplift_at_10pct": uplift_at_k(y, score, treatment, 0.10),  # 判断实际目标人群价值
        "uplift_at_20pct": uplift_at_k(y, score, treatment, 0.20),
        "ate_observed": float(y[treatment == 1].mean() - y[treatment == 0].mean()),
        "cate_mean": float(np.mean(score)),  # 只用于校准诊断
        "cate_std": float(np.std(score)),
    }
