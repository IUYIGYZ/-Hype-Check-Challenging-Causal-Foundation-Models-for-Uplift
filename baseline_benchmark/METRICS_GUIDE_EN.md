# `metrics.py` Code and Evaluation Metrics Guide

## 1. Purpose of the file

`baseline_benchmark/metrics.py` evaluates the performance of uplift/CATE models.

It does not load data, preprocess features, or train a model. It receives the CATE scores already predicted by a model and evaluates whether the model ranks users who are more likely to benefit from treatment near the top.

Its position in the baseline pipeline is:

```text
Processed X_train / X_val / X_test
                  ↓
          Train a model and predict CATE
                  ↓
       metrics.py evaluates the CATE ranking
                  ↓
       Qini, AUUC, Uplift@10%, Uplift@20%
```

---

## 2. Main inputs

The main entry point is:

```python
evaluate_uplift(y, score, treatment)
```

The three inputs must refer to the same observations in the same order:

| Input | Meaning | Current values |
|---|---|---|
| `y` | Observed outcome, such as whether a user purchased | `0` or `1` |
| `treatment` | Whether the user received the intervention | `0` or `1` |
| `score` | CATE/uplift predicted by the model | Any finite real number |

In this project:

```text
T = 1: the user received treatment
T = 0: the user did not receive treatment, i.e. control
Y = 1: the target event occurred, such as a purchase
Y = 0: the target event did not occur
```

`score` is the model's estimate of the Conditional Average Treatment Effect:

\[
\hat\tau(x)
=
\widehat P(Y=1\mid X=x,T=1)
-
\widehat P(Y=1\mid X=x,T=0)
\]

For example, `score = 0.08` for a user means:

> The model predicts that treatment increases the purchase probability of users with these characteristics by approximately 0.08, or 8 percentage points.

This is a model prediction. It is not an individually observed true treatment effect.

---

## 3. Why ordinary prediction accuracy is not the primary metric

There are two reasons.

### 3.1 The business objective is different

An ordinary purchase prediction model answers:

> Will this user make a purchase?

An uplift model answers:

> Will this user become more likely to purchase because of the treatment?

A user who would purchase anyway may be easy for a standard classifier to predict correctly, but sending that user a coupon may provide no incremental value.

Therefore, ordinary classification accuracy is not sufficient for deciding whom to treat.

### 3.2 The individual counterfactual is not jointly observable

For the same user, we can observe only one state:

- the outcome after receiving treatment; or
- the outcome without receiving treatment.

We cannot simultaneously observe both potential outcomes for the same user. Therefore, the true individual treatment effect is generally unobserved, and ordinary individual-level CATE prediction error cannot be calculated directly.

The metrics in this file instead use statistical differences between treatment and control groups to assess the quality of the predicted uplift ranking.

---

## 4. `_as_arrays`: input validation

`_as_arrays` converts the three inputs into one-dimensional NumPy arrays and verifies that:

1. `y`, `score`, and `treatment` have the same length;
2. the arrays are not empty;
3. none of the arrays contains `NaN` or infinity;
4. `y` is a binary outcome;
5. `treatment` is encoded as `0/1`;
6. both treatment and control observations are present.

If an evaluation set contains only one treatment arm, a treatment-control outcome difference cannot be calculated. The function therefore raises an error rather than returning a misleading metric.

After validation, treatment labels are stored as `int8` to keep their representation compact.

---

## 5. `_threshold_indices`: handling tied prediction scores

A model may assign exactly the same CATE score to multiple users. For example:

```text
0.20, 0.20, 0.20, 0.10, 0.10
```

`_threshold_indices` identifies the positions at which the score changes. Users with the same score are therefore processed as one group when constructing the Qini and uplift curves.

This prevents the curve from changing merely because observations with equal scores appear in a different row order.

For one observation, the only threshold is index zero. For multiple observations, the final observation is always included so that the curve reaches the full evaluation sample.

---

## 6. `qini_curve`: the Qini curve

The function first sorts users from the highest predicted CATE to the lowest.

For the top \(k\) ranked users, define:

- \(N_t(k)\): number of treated users;
- \(N_c(k)\): number of control users;
- \(Y_t(k)\): number of observed positive outcomes among treated users;
- \(Y_c(k)\): number of observed positive outcomes among control users.

The cumulative Qini gain is:

\[
Q(k)
=
Y_t(k)
-
Y_c(k)\frac{N_t(k)}{N_c(k)}
\]

It can be interpreted as:

> Among the top \(k\) ranked users, take the observed number of positive outcomes in treatment and subtract the expected number without treatment, estimated by scaling the control outcomes to the treated-group size.

For example:

```text
Top 1,000 ranked users:
Treatment: 500 users, 50 purchases
Control:   500 users, 30 purchases
```

Then:

\[
Q(1000)=50-30\times\frac{500}{500}=20
\]

Under this statistical estimate, treatment generated approximately 20 additional purchases in the selected population.

If an early section of the ranking contains no control observations, the denominator is zero. The code assigns a gain of zero at that point to avoid division by zero. This follows the behavior of the reference curve construction.

The returned arrays contain:

- `x`: the cumulative number of ranked users included;
- `gain`: cumulative Qini gain at each distinct score threshold.

The point `(0, 0)` is added so that the curve starts at the origin.

---

## 7. `qini_auc_normalized`: normalized Qini AUC

This metric compares three curves:

- the current model's Qini curve;
- the random-ranking baseline;
- an ideal Qini curve constructed from the observed `Y` and `T`.

It is calculated as:

\[
\text{Normalized Qini}
=
\frac{
AUC_{model}-AUC_{random}
}{
AUC_{perfect}-AUC_{random}
}
\]

Basic interpretation:

- greater than 0: the ranking is better than random overall;
- equal to 0: the ranking is equivalent to random or constant scores;
- less than 0: the ranking is poor and may be reversed;
- larger values are generally better.

The automatic tuning code in this project uses `qini_auc_normalized` as the default validation objective. This is appropriate because the principal goal is to rank users by incremental treatment effect.

### Important range note

`qini_auc_normalized` is not a probability and should not be assumed to lie strictly in `[0, 1]`.

With a very small sample, few positive outcomes, or treatment-control imbalance, empirical curve fluctuations can produce a value above 1 or below -1. This does not automatically indicate an implementation error, but it is a reason to inspect sample size and result stability.

If the ideal curve has essentially no area above the random baseline, the denominator is too small for meaningful normalization. The implementation returns `NaN` instead of dividing by a value near zero.

---

## 8. `qini_coefficient`: unnormalized Qini area

This implementation calculates:

\[
\frac{
AUC_{model}-AUC_{random}
}{N^2}
\]

Dividing by \(N^2\) reduces the direct effect of evaluation sample size on the numerical scale.

The difference from `qini_auc_normalized` is:

- `qini_auc_normalized` additionally divides by the ideal curve's area above the random baseline;
- `qini_coefficient` only subtracts the random baseline and scales by sample size.

The coefficient can be used to compare models on exactly the same dataset and evaluation split. It should not be used alone to compare performance across different datasets because outcome rates and treatment proportions may differ.

A positive coefficient indicates area above the random-ranking line. A negative coefficient indicates that much of the ranking curve lies below the random baseline.

---

## 9. `uplift_curve`: the uplift gain curve

For the top \(k\) ranked users, the function calculates:

\[
U(k)
=
k\left(
\frac{Y_t(k)}{N_t(k)}
-
\frac{Y_c(k)}{N_c(k)}
\right)
\]

The expression inside the parentheses is:

```text
treated-group outcome rate - control-group outcome rate
```

For example:

```text
Treatment purchase rate = 10%
Control purchase rate   = 6%
```

The observed uplift in that population is 4 percentage points.

Multiplication by \(k\) converts the rate difference into a cumulative gain scale.

Qini and uplift curves both assess the CATE ranking, but they use different scaling:

- Qini gain is primarily scaled to the number of treated observations;
- uplift gain multiplies the treatment-control rate difference by the total number of selected users.

If one arm has not yet appeared in an early part of the ranking, its response rate is set to zero by safe division. The curve is also prefixed with `(0, 0)`.

---

## 10. `_perfect_uplift_curve`: ideal uplift curve

This function constructs an ideal ranking score using the observed `Y` and `T`. The resulting curve is used as the reference curve for normalized uplift AUC.

The construction distinguishes observed combinations such as:

- treated responders;
- control non-responders;
- control responders;
- treated non-responders.

The exact ordering depends on the relative number of control responders and treated non-responders, following the scikit-uplift definition.

This is not a prediction from any baseline model and is never used to train the model. It is only an evaluation-time reference curve.

---

## 11. `uplift_auc_normalized`: normalized uplift AUC

The calculation is:

\[
\text{Normalized Uplift AUC}
=
\frac{
AUC_{model}-AUC_{random}
}{
AUC_{perfect}-AUC_{random}
}
\]

It summarizes the whole uplift curve instead of examining only the top 10% or top 20% of users.

Basic interpretation:

- larger values are generally better;
- a value near 0 indicates performance close to random ranking;
- a negative value indicates a poor ranking.

It is useful as a secondary metric for checking whether the uplift-curve conclusion agrees with the primary normalized Qini result.

If the ideal uplift curve has no meaningful area above the baseline, the function returns `NaN` because the normalized score would not be well-defined.

---

## 12. `uplift_at_k`: observed uplift in the targeted population

`uplift_at_k` sorts users from highest to lowest predicted CATE and selects only the top fraction.

For example:

```python
uplift_at_k(y, score, treatment, 0.10)
```

calculates:

\[
\text{Uplift@10\%}
=
\bar Y_{treatment,top10\%}
-
\bar Y_{control,top10\%}
\]

Suppose that among the top 10% ranked users:

```text
Treatment purchase rate = 3.2%
Control purchase rate   = 2.1%
```

Then:

```text
Uplift@10% = 0.032 - 0.021 = 0.011
```

The observed treatment-group purchase rate is therefore 1.1 percentage points higher than the control-group rate among the model's top-ranked users.

The project reports:

- `uplift_at_10pct`: useful when the budget allows treatment of only the top 10%;
- `uplift_at_20pct`: useful when the budget allows treatment of the top 20%.

These metrics are closely connected to a practical targeting policy:

> If only a limited fraction of users can be treated, does the model identify a population with a strong observed incremental response?

### Boundary cases

The number of selected observations is:

```python
k = max(1, int(len(y) * fraction))
```

The fractional cutoff is therefore rounded down, while at least one observation is retained for very small evaluation samples.

If the selected top 10% or top 20% contains only treatment or only control observations, the treatment-control rate difference cannot be computed. The function returns `NaN`.

This case is more likely in small test samples. It should be uncommon in a large randomized evaluation split, but it must still be checked.

If multiple users have equal CATE scores exactly at the cutoff, the current implementation selects observations according to their original stable row order. As a result, Uplift@K can change slightly with row order in a cutoff tie.

For this reason, Uplift@K should be treated as a business-oriented secondary metric, while whole-curve Qini AUC remains the primary model-selection metric.

---

## 13. `auuc`: raw area under the uplift curve

The current implementation defines:

\[
AUUC
=
\frac{AUC_{uplift\ curve}}{N^2}
\]

It does not subtract the random-ranking baseline.

Therefore:

> AUUC for random ranking or constant CATE predictions is not necessarily zero.

If the experiment has a positive overall average treatment effect, even a random user ordering can produce a positive area under the uplift curve.

This differs from `qini_coefficient`, which explicitly subtracts the random-ranking line.

Within the same dataset and evaluation split, all models share the same users, treatment assignment, outcomes, and overall observed effect. AUUC can therefore still be used as a secondary comparison metric.

It should not be interpreted as a calibrated probability or directly compared across unrelated datasets.

---

## 14. `evaluate_uplift`: unified evaluation entry point

`evaluate_uplift` returns a dictionary with nine values:

| Metric | Main purpose | Is larger better? |
|---|---|---:|
| `qini_auc_normalized` | Primary uplift-ranking metric and current tuning objective | Yes |
| `qini_coefficient` | Qini area above the random-ranking baseline | Yes |
| `uplift_auc_normalized` | Whole uplift-curve ranking quality | Yes |
| `auuc` | Raw area under the uplift curve | Within the same dataset, yes |
| `uplift_at_10pct` | Observed uplift among the top 10% | Yes |
| `uplift_at_20pct` | Observed uplift among the top 20% | Yes |
| `ate_observed` | Overall treatment-control outcome-rate difference | Not a model-ranking metric |
| `cate_mean` | Mean predicted CATE | No |
| `cate_std` | Standard deviation of predicted CATE | No |

The first six values evaluate or summarize model ranking behavior. The final three values are mainly diagnostics.

---

## 15. `ate_observed`: observed overall group difference

The function calculates:

\[
\text{ATE}_{observed}
=
\bar Y_{T=1}
-
\bar Y_{T=0}
\]

This is the average outcome-rate difference between treatment and control in the complete evaluation split.

In randomized experimental data, it is a simple estimate of the overall average treatment effect.

However:

- it is calculated directly from `Y` and `T`;
- it does not use the model's predicted `score`;
- every model evaluated on the same split has the same `ate_observed`.

It therefore cannot be used to select a model or tune hyperparameters.

For non-randomized observational data, treated and control users may be systematically different. In that setting, the raw difference cannot automatically be interpreted as a causal ATE. Adjustment methods such as propensity weighting may be required.

The name `ate_observed` intentionally emphasizes that this is an observed group difference, not a directly observed individual treatment effect.

---

## 16. `cate_mean` and `cate_std`: prediction-distribution diagnostics

### `cate_mean`

```python
np.mean(score)
```

This is the average CATE predicted by the model.

It can be compared roughly with `ate_observed`:

- if they are reasonably close, the model's average effect prediction may be plausible;
- if they are very different, the model may have a substantial overall calibration bias.

Agreement does not prove that the user ranking is correct. A model can predict the correct average effect while assigning effects to the wrong users.

`cate_mean` is not a metric for which a larger value is automatically better.

### `cate_std`

```python
np.std(score)
```

This measures how much predicted treatment effects differ across users.

- a value close to zero means that the model gives almost every user the same score;
- a larger value means that the model predicts stronger treatment-effect heterogeneity;
- an extremely large value may also indicate overfitting or unstable, extreme predictions.

`cate_std` is diagnostic information, not a performance measure. A larger standard deviation does not by itself mean that the model is better.

---

## 17. Recommended use for tuning and model selection

For T-Learner, X-Learner, DR-Learner, and DragonNet, the recommended priority is:

1. use validation `qini_auc_normalized` as the primary tuning objective;
2. check whether `uplift_auc_normalized` supports a similar conclusion;
3. use `uplift_at_10pct` and `uplift_at_20pct` to assess targeting value under limited intervention budgets;
4. use `cate_mean`, `cate_std`, and `ate_observed` to identify clearly abnormal prediction distributions;
5. after hyperparameter selection is complete, evaluate the selected model once on the independent test split;
6. do not repeatedly adjust hyperparameters based on test results.

A slightly higher validation Qini in a single trial is not sufficient evidence that one model is truly superior. In low-conversion data, a small number of positive outcomes can cause noticeable metric variation.

When deciding whether tuning can stop, examine all of the following:

- whether additional trials have stopped producing stable improvements;
- whether the primary and secondary metrics point in the same direction;
- whether results are stable across multiple random seeds or bootstrap samples;
- whether the apparent improvement is larger than expected statistical variation;
- whether the additional computational cost is justified by the observed gain.

The test split should remain untouched during this process. Validation chooses parameters; test provides the final report.

---

## 18. Assumptions and limitations

### 18.1 Binary outcomes only

`metrics.py` currently requires `Y` to be `0/1`.

It can directly evaluate binary outcomes such as `conversion` and `visit`, but it cannot directly evaluate a continuous outcome such as `spend`.

A continuous-outcome extension would require revisiting the validation logic and confirming that every curve and ideal-ranking definition remains appropriate.

### 18.2 Binary treatment only

`T` must be `0/1`.

If the project later compares multiple email types, coupon types, or treatment doses simultaneously, multi-treatment metrics will be required.

### 18.3 Causal interpretation requires randomization or identification assumptions

When treatment is randomized, treatment-control differences within ranked groups have a straightforward experimental interpretation, subject to sampling variation.

When treatment is not randomized, the difference can contain confounding bias. In that case, the simple group comparisons in this file are not sufficient by themselves. Methods such as propensity scores, inverse-probability weighting, or doubly robust policy evaluation may be needed.

The fact that a DR-Learner estimates treatment effects does not automatically make an unadjusted evaluation metric unbiased. Estimation and evaluation are separate issues.

### 18.4 No confidence intervals are currently reported

The current functions return point estimates only. They do not produce bootstrap standard errors or confidence intervals.

Therefore, if model A has a Qini score only 0.01 higher than model B, that does not necessarily prove that A is truly better. Confidence intervals, repeated seeds, and stability analyses belong to the project's later robustness-testing stage.

### 18.5 Exact sorting cost

Curve metrics sort all evaluation scores and therefore require approximately \(O(N\log N)\) sorting time.

This is reasonable for the current benchmark and provides exact ranking curves, but it should be remembered when evaluating millions of Criteo rows.

---

## 19. Code review conclusion

The current implementation was checked with the following conclusions:

- the core Qini curve formula is correct;
- the core uplift curve formula is correct;
- the random-baseline area is handled correctly;
- normalized Qini AUC and normalized uplift AUC follow the scikit-uplift calculation structure;
- tied CATE scores are grouped when constructing full curves;
- length, finiteness, binary-label, and treatment-arm checks are present;
- degenerate normalization denominators return `NaN` rather than unsafe values;
- the complete project test result was `23 passed, 1 skipped`.

No core formula problem was found that would directly invalidate the comparison among T-Learner, X-Learner, DR-Learner, and DragonNet. The file can continue to be used for baseline tuning and evaluation.

The main points requiring careful interpretation are:

1. normalized Qini can fluctuate strongly in small or low-conversion samples;
2. Uplift@K returns `NaN` if the selected subset lacks one treatment arm;
3. AUUC does not subtract the random baseline;
4. `ate_observed`, `cate_mean`, and `cate_std` are not ranking-performance metrics;
5. the current output does not include uncertainty estimates or confidence intervals;
6. simple treatment-control differences require randomized treatment, or suitable causal adjustment, for a causal interpretation.

