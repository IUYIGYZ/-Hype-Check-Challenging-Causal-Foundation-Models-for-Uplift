# CausalPFN Uplift Baseline Benchmark

## Purpose

This directory implements a runnable baseline comparison for binary-treatment, binary-outcome uplift experiments. Every model receives the same cleaned samples, split, transformed feature matrix, treatment labels, outcomes, and test-set metrics.

CausalPFN is available as an optional pretrained foundation-model baseline. It uses the same transformed train/test matrices and its CATE predictions are passed through the existing `evaluate_uplift` pipeline.

## Implemented Models

- `constant_ate`: constant-score sanity check; it has no individual ranking ability.
- `s_learner`: one outcome model for `E[Y|X,T]`.
- `t_learner`: separate outcome models for treated and control samples.
- `x_learner`: imputed-effect learner designed to work well when treatment-arm sizes are imbalanced.
- `dr_learner`: cross-fitted doubly robust pseudo-outcome followed by an effect regression.
- `tarnet`: shared neural representation with two potential-outcome heads.
- `dragonnet`: TARNet-style outcome heads plus a propensity head.
- `causalpfn`: pretrained CausalPFN transformer, used through the official `CATEEstimator` API. It performs in-context causal effect estimation and does not train a new task-specific model.

The current DragonNet implementation is the basic architecture and does **not** implement targeted regularization. Report it as `DragonNet (basic, no targeted regularization)`. Use the authors' implementation or CATENets for strict paper-level reproduction.

## References

Local project references:

- `../Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf`
- `../参考文献/2506.07918v2.pdf`
- `../参考文献/2605.26288v1.pdf`
- `../参考文献/2605.27473v1.pdf`

Primary and official external sources:

- S/T/X meta-learners: https://doi.org/10.1073/pnas.1804597116
- DR-Learner: https://arxiv.org/abs/2004.14497
- TARNet/CFR: https://proceedings.mlr.press/v70/shalit17a.html
- DragonNet: https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN repository: https://github.com/vdblm/CausalPFN
- EconML documentation: https://econml.azurewebsites.net/
- CATENets repository: https://github.com/AliciaCurth/CATENets
- scikit-uplift Qini definition: https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html

## Fair-Comparison Protocol

The data pipeline runs once per dataset and seed:

1. load aligned `X/T/Y` from the same cleaned Parquet files;
2. fix one primary outcome;
3. create one train/validation/test split;
4. fit imputation, scaling, and categorical encoding on the training split only;
5. transform the three splits once and share the resulting matrices across all models;
6. predict CATE on identical test rows;
7. evaluate all predictions with the same Qini/AUUC functions.

Criteo and LZD use a full-feature-vector hash as a grouping key so duplicate feature vectors cannot cross split boundaries. Hillstrom and RetailHero use stratification by `T × Y`.

The model feature matrix excludes `epk_id`, `T`, `treatment_dt`, `split`, outcome-file IDs, lag fields, and non-selected outcomes.

## Structure

```text
baseline_benchmark/
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

## Environment

Traditional models run in the default Python environment. The default environment's PyTorch has a CUDA shared-library conflict on this machine. The existing `Torch25` Conda environment was verified for both traditional and neural models.

Run commands from this directory:

```bash
cd 项目/baseline_benchmark
```

## Quick Runs

Traditional baselines:

```bash
python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner \
  --max-rows 5000 \
  --tree-max-iter 20 \
  --seed 17
```

Neural baselines:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models tarnet,dragonnet \
  --max-rows 5000 \
  --epochs 20 \
  --seed 17
```

CausalPFN (optional dependency):

```bash
pip install -r requirements-causalpfn.txt
python run_baselines.py \
  --dataset retailhero \
  --models causalpfn \
  --max-rows 5000 \
  --device auto \
  --seed 17
```

`causalpfn` can be combined with any existing model name, for example
`--models s_learner,t_learner,tarnet,dragonnet,causalpfn`. The benchmark writes
the same `metrics.csv` and `predictions.parquet` fields for every model, so Qini,
AUUC, uplift@10%/20%, timing, and the multi-seed suite remain directly comparable.

Optional training-only treatment upsampling:

```bash
python run_baselines.py --dataset lzd --models constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet,causalpfn --upsample-train
```

`--upsample-train` balances the two treatment arms after splitting, by sampling
training rows with replacement. Validation and test rows are never duplicated;
the manifest records both `n_train` and `n_train_fit`.

All models:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet \
  --max-rows 50000 \
  --epochs 100 \
  --seed 0
```

Set `--max-rows 0` to use all cleaned rows.

## Multi-Seed Suite

Short verification:

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1 \
  --max-rows 10000 \
  --epochs 20 \
  --tree-max-iter 50
```

Ten-seed experiment:

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-rows 50000 \
  --epochs 100 \
  --tree-max-iter 150
```

Do not start with full Criteo. Conversion is extremely sparse, and a uniform 50k subsample can contain very few control-arm conversions. Establish the protocol on RetailHero, LZD, and Hillstrom first, then design a separate Criteo sample-size and uncertainty protocol.

## Outputs

A single run produces:

```text
metrics.csv
predictions.parquet
splits.parquet
transformed_features.csv
run_config.json
```

`predictions.parquet` contains:

```text
epk_id, dataset, outcome, model, seed, T, Y, cate_pred
```

A suite additionally produces `all_metrics.csv`, `summary.csv`, and `suite_config.json`. The summary contains the mean, standard deviation, and normal-approximation 95% half-width across seeds.

## Metrics

The curve and normalized-score calculations follow the public scikit-uplift
implementation (`sklift.metrics.metrics.py`) so results are comparable with
the library's Qini/uplift conventions.

- normalized Qini AUC using the scikit-uplift convention;
- Qini area above the random-ranking line, scaled by `N²`;
- AUUC over the top-fraction outcome-rate difference curve;
- uplift@10% and uplift@20%;
- raw test-set ATE;
- mean and standard deviation of CATE predictions;
- fitting and prediction time.

Real marketing RCTs do not provide unit-level true CATE, so PEHE is not computed here. PEHE should be added only for semi-synthetic datasets such as IHDP or ACIC with known effects.

## Current Boundaries

1. This is a runnable and auditable first baseline version, not a line-by-line reproduction of every EconML/CATENets default.
2. S/T/X/DR use the same scikit-learn histogram gradient boosting base learner to control capacity fairly.
3. DR uses cross-fitting; S/T/X are evaluated on an independent test split.
4. CausalPFN is a pretrained foundation model and may download model weights on first use. Results should record the installed package/version and whether CPU or CUDA was used.
5. Causal Forest is not included because EconML is not installed. Add `CausalForestDML` or GRF for the final paper-level benchmark.
6. The reported interval is currently across random seeds. Test-set bootstrap confidence intervals should still be added for the final protocol.
7. Orange Telecom is excluded because the treatment provenance and cleaned metadata remain causally ambiguous.
8. The cleaned Hillstrom data combine Men's and Women's email into `T=1`. This estimates “any marketing email vs no email,” not the paper's separate `Hill(1)` and `Hill(2)` tasks. Reconstruct two binary cohorts from the raw `segment` field for strict reproduction.

