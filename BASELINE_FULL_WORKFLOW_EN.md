# Complete Baseline Workflow for the CausalPFN Project (English)

## 1. Scope and Objective

This document describes the completed baseline work: cleaned-data organization, common splitting and preprocessing, classical and neural CATE estimators, uplift metrics, repeated experiments, saved artifacts, code-review results, and known limitations.

Implementation: `项目/baseline_benchmark/`.

CausalPFN is not connected yet. A later CausalPFN adapter only needs to predict `cate_pred` for the same test IDs and use the same evaluation functions.

Given pre-treatment features `X`, binary treatment `T`, and binary outcome `Y`, the estimand is:

\[
\tau(x)=E[Y(1)-Y(0)\mid X=x].
\]

Real marketing RCTs do not reveal unit-level true CATE, so the primary objective is uplift ranking rather than ordinary classification accuracy.

## 2. Dataset Organization

The benchmark reads `项目/data/data_A_cleaned/`.

| Dataset | Rows | Default outcome | Current preparation | Recommendation |
|---|---:|---|---|---|
| Criteo | 13,979,592 | `conversion` | Redundant features removed; duplicate vectors require group-safe splitting | Usable, but extremely sparse |
| Hillstrom | 64,000 | `conversion` | Text categories retained and encoded at runtime | Usable, low conversion |
| LZD | 181,669 | `Y` | Only randomized test partition retained; redundant features removed | Usable |
| RetailHero | 200,039 | `Y` | Labeled rows retained; suspected post-treatment fields removed | Usable |
| Orange Telecom | 11,896 | `churn` | Treatment provenance conflicts with metadata | Excluded from primary experiments |

The cleaned Hillstrom data combine Men's and Women's email as `T=1`; the estimand is therefore “any marketing email vs no email.” Strict reproduction of CausalPFN's `Hill(1)` and `Hill(2)` requires two cohorts constructed from the raw `segment` field.

## 3. Data Pipeline

### 3.1 Loading and Validation

`baseline_benchmark/data.py`:

1. reads `features.parquet` and `outcomes.parquet`;
2. validates equal row counts and row-wise `epk_id` alignment;
3. verifies that treatment contains both 0 and 1 and that the selected outcome is binary;
4. reads selected rows in PyArrow batches for large datasets;
5. rejects non-finite final model matrices.

Default outcomes:

```yaml
criteo: conversion
hillstrom: conversion
lzd: Y
retailhero: Y
```

Excluded from `X`:

```text
epk_id, T, treatment_dt, split, lag, and non-selected outcomes
```

`T` remains a separate treatment input. Criteo `exposure` is post-treatment and is not used as a feature or primary outcome.

### 3.2 Optional Subsampling

`--max-rows N` draws N rows without replacement using the run seed. `--max-rows 0` uses all rows. Every model in a run shares the identical sampled cohort.

Criteo conversion is so sparse that a uniform 50k sample can contain very few control conversions. Formal Criteo experiments need a larger sample and bootstrap intervals.

### 3.3 Train/Validation/Test Split

Default proportions:

```text
train 60% / validation 20% / test 20%
```

- Hillstrom and RetailHero: `StratifiedShuffleSplit` on `T × Y`.
- Criteo and LZD: `GroupShuffleSplit` using a hash of the full raw feature vector, preventing duplicate feature vectors from crossing split boundaries.

Post-split checks enforce a disjoint and complete partition and both treatment arms in every split. A warning is emitted when any arm has fewer than ten positive outcomes.

### 3.4 Categorical, Missing-Value, and Numerical Processing

Columns with `object`, `category`, or `string` dtype use:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

Hillstrom `history_segment`, `zip_code`, and `channel` are converted to numeric one-hot columns. An unseen test category does not refit or change the feature dimension.

Numerical columns use:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

The split is created first. Imputation, scaling, and encoding are fitted on `X_train` only, then applied to train/validation/test. This prevents preprocessing leakage.

All estimators share the same arrays:

```text
X_train/T_train/Y_train
X_validation/T_validation/Y_validation
X_test/T_test/Y_test
```

## 4. Baseline Models

Classical methods share scikit-learn histogram gradient boosting base learners to reduce capacity differences.

### Constant ATE

Outputs the training-set `mean(Y|T=1)-mean(Y|T=0)` for every test row. It has no ranking ability and is a metric sanity check; normalized Qini and Uplift AUC should be zero.

### S-Learner

Fits one response model `mu(X,T)`:

\[
\hat\tau(x)=\hat\mu(x,1)-\hat\mu(x,0).
\]

### T-Learner

Fits separate `mu0(X)` and `mu1(X)` models:

\[
\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x).
\]

### X-Learner

Builds `D0=mu1(X0)-Y0` and `D1=Y1-mu0(X1)`, learns `tau0/tau1`, and combines them with empirical propensity `g=P(T=1)`:

\[
\hat\tau(x)=g\hat\tau_0(x)+(1-g)\hat\tau_1(x).
\]

This is an important baseline under treatment-arm imbalance.

### DR-Learner

Stratified K-fold cross-fitting produces out-of-fold `mu0/mu1`. For these RCTs, empirical constant propensity `e` is used:

\[
\tilde\tau=\hat\mu_1-\hat\mu_0+
\frac{T(Y-\hat\mu_1)}{e}-
\frac{(1-T)(Y-\hat\mu_0)}{1-e}.
\]

The final model regresses this pseudo-outcome on `X`. Fitting fails clearly if any `T × Y` stratum has fewer than two samples.

### TARNet

A PyTorch shared representation with control and treated outcome heads. Binary cross-entropy is evaluated only on the observed head, with validation-loss early stopping.

### DragonNet

Adds a propensity head and optimizes outcome BCE plus treatment BCE. This implementation must be reported as `DragonNet (basic, no targeted regularization)`. Strict reproduction requires the authors' code or CATENets.

Neural reproducibility settings include NumPy/PyTorch seeds, deterministic algorithms, deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Neural runs use the verified `Torch25` Conda environment.

## 5. Metrics

All models call `baseline_benchmark/metrics.py`:

- `qini_auc_normalized`: compatible with scikit-uplift `qini_auc_score`;
- `qini_coefficient`: Qini area above random, scaled by `N²`;
- `uplift_auc_normalized`: compatible with scikit-uplift `uplift_auc_score`;
- `auuc`: raw standard uplift-gain area scaled by `N²`;
- `uplift_at_10pct` and `uplift_at_20pct`;
- raw test ATE, CATE mean/standard deviation, fit and prediction time.

Equal predicted scores are grouped at one threshold so their internal row order cannot alter Qini/Uplift AUC. Uplift@k returns NaN if the selected top group lacks either treatment arm.

PEHE is not reported for real campaigns because true unit-level effects are unavailable. It should be added when IHDP/ACIC are connected.

## 6. Code and Commands

```text
项目/baseline_benchmark/
├── baseline_benchmark/
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   └── neural.py
├── tests/test_smoke.py
├── run_baselines.py
├── run_suite.py
└── requirements.txt
```

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark
```

Traditional smoke run:

```bash
python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner \
  --max-rows 5000 --tree-max-iter 20 --seed 17
```

Neural run:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero --models tarnet,dragonnet \
  --max-rows 5000 --epochs 20 --seed 17
```

All baselines:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet \
  --max-rows 50000 --epochs 100 --tree-max-iter 150 --seed 0
```

Add `--save-transformed-data` to export final numeric train/validation/test Parquet files for later CausalPFN reuse.

Ten-seed suite:

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-rows 50000 --epochs 100 --tree-max-iter 150
```

## 7. Output Organization

```text
results/<dataset>/seed_<seed>_<timestamp>/
├── data_manifest.json
├── metrics.csv
├── predictions.parquet
├── preprocessor.joblib
├── run_config.json
├── splits.parquet
├── transformed_features.csv
└── prepared_data/             # with --save-transformed-data
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

- `data_manifest.json`: path, outcome, seed, group-safe flag, feature names, and split/arm/event counts;
- `preprocessor.joblib`: fitted imputer, scaler, one-hot categories, and column order;
- `splits.parquet`: `epk_id, split`;
- `predictions.parquet`: `epk_id,dataset,outcome,model,seed,T,Y,cate_pred`;
- `metrics.csv`: one row per model;
- `prepared_data/*.parquet`: final numeric features plus `epk_id/T/Y`.

`run_suite.py` additionally creates `all_metrics.csv`, `summary.csv`, and `suite_config.json`. The summary reports mean, standard deviation, and a normal-approximation 95% half-width across seeds.

## 8. Fair-Comparison Guarantees

1. identical sampled cohort and split;
2. identical transformed `X/T/Y`;
3. identical test IDs;
4. identical metric functions;
5. training-only preprocessing fit;
6. saved split, preprocessor, configuration, predictions, and metrics.

This first version uses fixed hyperparameters. A formal main table still needs a predeclared, comparable validation-set tuning budget, with the test set reserved for final evaluation.

## 9. Code Review and Validation

Review covered data loading, ID alignment, stratified/group-safe splitting, unseen categories, train-only preprocessing, S/T/X/DR formulas, DR cross-fitting, neural training, official Qini/Uplift definitions, output writing, and CUDA reproducibility.

Fixes made during review:

1. added standard `uplift_auc_normalized` and clarified raw `auuc`;
2. added complete/disjoint split, treatment-arm, and low-event checks;
3. added minimum-stratum validation for DR cross-fitting;
4. added `preprocessor.joblib`, `data_manifest.json`, and optional numeric-data export;
5. added CUDA/cuBLAS determinism configuration;
6. fixed NumPy 1.23 strata-string compatibility and static formatting issues.

Final validation:

```text
pytest: 5 passed
flake8: 0 errors
mypy: 0 errors
syntax compilation: passed
traditional end-to-end run: passed
TARNet/DragonNet in Torch25: passed
prepared-data export: passed
```

Two pytest warnings originate from the environment's numexpr/distutils deprecation, not from this code. Smoke-test metrics are not scientific results.

## 10. Current Limitations and Next Steps

- CausalPFN is not connected yet; reuse the same split/numeric data and metrics next.
- Causal Forest is absent because EconML is not installed; add CausalForestDML or GRF for the final main table.
- DragonNet does not include targeted regularization.
- Formal validation-set hyperparameter search is not implemented.
- Current intervals summarize seeds; final experiments still need test-set bootstrap 95% CIs.
- Criteo needs a larger and dedicated low-conversion protocol.
- Q-Learner and GP-CATE belong to later robustness tests; CausalPFN-Rank follows only after locating default-model failures.

## 11. References

Local: `Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf` and `参考文献/2506.07918v2.pdf`, `2605.26288v1.pdf`, `2605.27473v1.pdf`.

Primary/official external sources:

- S/T/X: https://doi.org/10.1073/pnas.1804597116
- DR-Learner: https://arxiv.org/abs/2004.14497
- TARNet/CFR: https://proceedings.mlr.press/v70/shalit17a.html
- DragonNet: https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN: https://github.com/vdblm/CausalPFN
- EconML: https://econml.azurewebsites.net/
- CATENets: https://github.com/AliciaCurth/CATENets
- scikit-uplift metrics: https://www.uplift-modeling.com/en/latest/_modules/sklift/metrics/metrics.html
