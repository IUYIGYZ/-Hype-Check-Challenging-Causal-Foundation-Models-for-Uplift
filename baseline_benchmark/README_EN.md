# CausalPFN Uplift Baseline Benchmark

## Purpose

This directory implements a runnable baseline comparison for binary-treatment, binary-outcome uplift experiments. Every model receives the same cleaned samples, split, transformed feature matrix, treatment labels, outcomes, and test-set metrics.

CausalPFN is integrated through the official `causalpfn` package. It receives the
same transformed training matrix and produces `cate_pred` for the same held-out
IDs as every baseline, so all existing Qini/AUUC calculations remain unchanged.

## Implemented Models

- `t_learner`: separate outcome models for treated and control samples.
- `x_learner`: imputed-effect learner designed to work well when treatment-arm sizes are imbalanced.
- `dr_learner`: cross-fitted doubly robust pseudo-outcome followed by an effect regression.
- `dragonnet`: a shared representation, two potential-outcome heads, and a propensity head.
- `causalpfn`: the official pretrained CATE estimator, used without task-specific tuning.
- `causalpfn_head_ft`: an experimental domain-adapted CausalPFN that freezes the
  backbone and fine-tunes only the prediction head against train-only,
  cross-fitted doubly robust potential-outcome pseudo-labels.
- `causalpfn_ridge_correction`: zero-shot CausalPFN plus a regularized linear
  correction trained on cross-fitted DR residuals.
- `causalpfn_hgb_correction`: zero-shot CausalPFN plus a low-capacity histogram
  gradient-boosting correction trained on the same residual target.
- `causalpfn_x_learner`: cross-fitted CausalPFN potential-outcome imputation
  followed by a continuous-outcome CausalPFN that supplies both X-Learner
  effect functions.

The first CausalPFN run downloads the official `vdblm/causalpfn` checkpoint from
Hugging Face unless `--causalpfn-model-path` points to a local checkpoint. Its
default context/query limits are 4096 and can be changed with the corresponding
`--causalpfn-*` command-line options. On small samples, the adapter safely clips
the FAISS neighbour count to the smaller treatment arm.

The current DragonNet implementation is the basic architecture and does **not** implement targeted regularization. Report it as `DragonNet (basic, no targeted regularization)`. Use the authors' implementation or CATENets for strict paper-level reproduction.

## References

Local project references:

- `../Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf`
- `../参考文献/2506.07918v2.pdf`
- `../参考文献/2605.26288v1.pdf`
- `../参考文献/2605.27473v1.pdf`

Primary and official external sources:

- T/X meta-learners: https://doi.org/10.1073/pnas.1804597116
- DR-Learner: https://arxiv.org/abs/2004.14497
- DragonNet: https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN repository: https://github.com/vdblm/CausalPFN
- CATENets repository: https://github.com/AliciaCurth/CATENets
- scikit-uplift Qini definition: https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html

## Fair-Comparison Protocol

The data pipeline runs once per dataset and seed:

1. load aligned `X/T/Y` from the same cleaned Parquet files;
2. fix one primary outcome;
3. create one train/validation/test split;
4. fit imputation, scaling, and categorical encoding on the training split only;
5. transform the three splits once and share the resulting matrices across all models;
6. predict CATE on identical validation rows while tuning, then on identical test rows after freezing parameters;
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

`run_baselines.py` and `run_suite.py` use `--evaluation-split validation` by default for tuning and deciding when to stop. After hyperparameters are frozen, final evaluation must explicitly use:

```text
--evaluation-split test
```

## Quick Runs

Traditional baselines:

```bash
python run_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner \
  --max-rows 5000 \
  --tree-max-iter 20 \
  --seed 17
```

Neural baselines:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models dragonnet \
  --max-rows 5000 \
  --epochs 20 \
  --seed 17
```

All models:

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet,causalpfn \
  --max-rows 50000 \
  --epochs 100 \
  --seed 0
```

Set `--max-rows 0` to use all cleaned rows.

## Experimental CausalPFN Head-Only Fine-Tuning

`causalpfn_head_ft` is deliberately separate from the zero-shot `causalpfn`
baseline. It performs the following operations using the outer training split
only:

1. cross-fit treatment-arm outcome models and construct AIPW/DR signals for
   both potential outcomes;
2. cross-fit a second regression stage to smooth those signals into bounded
   `E[Y(0)|X]` and `E[Y(1)|X]` pseudo-labels;
3. create an inner train/validation split;
4. freeze the official CausalPFN backbone and optimize only
   `icl_model.model.head`;
5. use fixed inner validation tasks for early stopping.

The outer validation or test outcomes are never used to create pseudo-labels or
update the model. Therefore validation remains suitable for selecting the
fine-tuning configuration, and test must remain sealed until that configuration
is frozen.

Example validation experiment:

```bash
python run_baselines.py \
  --cleaned-root "path/to/data_A_cleaned" \
  --dataset hillstrom \
  --models causalpfn,causalpfn_head_ft \
  --evaluation-split validation \
  --seed 42 \
  --causalpfn-ft-epochs 10 \
  --causalpfn-ft-learning-rate 1e-4 \
  --causalpfn-ft-context-length 1024 \
  --causalpfn-ft-query-length 256 \
  --causalpfn-ft-tasks-per-epoch 8 \
  --causalpfn-pseudo-folds 5
```

In addition to the normal benchmark artifacts, the fine-tuned run writes
`causalpfn_head_ft_training.json`. It records DR cross-fitting diagnostics,
per-epoch training/inner-validation loss, early-stopping state, and the numbers
of trainable and frozen parameters. Report this model as domain-adapted
CausalPFN, not as zero-shot CausalPFN.

## Experimental CausalPFN Residual Correction

The correction experiments leave the official CausalPFN checkpoint unchanged.
For every outer-training row they construct:

1. an OOF CausalPFN CATE prediction from a context that excludes that row's
   fold;
2. OOF treatment-arm outcome predictions and an AIPW/DR effect;
3. a winsorized residual target `DR effect - OOF CausalPFN CATE`;
4. correction features `[X, CATE_PFN, mu0, mu1]`.

The Ridge or HGB model learns that residual. Final predictions use:

```text
CATE_final = CATE_PFN + correction_strength * predicted_residual
```

Example:

```bash
python run_baselines.py \
  --cleaned-root "path/to/data_A_cleaned" \
  --dataset lzd \
  --models causalpfn,causalpfn_ridge_correction,causalpfn_hgb_correction \
  --evaluation-split validation \
  --causalpfn-correction-folds 3 \
  --causalpfn-correction-strength 0.5 \
  --causalpfn-correction-ridge-alpha 10 \
  --causalpfn-correction-max-iter 50 \
  --causalpfn-correction-max-leaf-nodes 15
```

The OOF CausalPFN pass is intentionally strict and therefore expensive: with
three folds it fits three fold-specific contexts and one final full-training
context per correction model. Each run writes
`<model_name>_training.json` with residual, OOF, and evaluation-correction
diagnostics. Select correction hyperparameters on validation only and keep test
sealed.

## Multi-Seed Validation and Final Evaluation

### Suggested tuning stopping rule

Read validation results only during tuning. Use `qini_auc_normalized` as the primary metric and `uplift_at_10pct` as the business-facing secondary metric. Screen settings with one seed, then verify the best two or three settings with at least three seeds.

Stop tuning when a new setting improves mean validation Qini by less than 0.005, or by less than the seed-level standard error, while Uplift@10% does not materially improve. Prefer the simpler and faster setting when performance is effectively tied. Evaluate test data only after freezing hyperparameters.


Short verification:

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1 \
  --max-rows 10000 \
  --epochs 20 \
  --tree-max-iter 50
```

Use validation to compare settings. After freezing hyperparameters, run the ten-seed test experiment:

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --evaluation-split test \
  --max-rows 50000 \
  --epochs 100 \
  --tree-max-iter 150
```

Do not start with full Criteo. Conversion is extremely sparse, and a uniform 50k subsample can contain very few control-arm conversions. Establish the protocol on RetailHero, LZD, and Hillstrom first, then design a separate Criteo sample-size and uncertainty protocol.

## Outputs

A single run is written under `results/<dataset>/<evaluation_split>_seed_<seed>_<timestamp>/` and produces:

```text
metrics.csv
predictions.parquet
splits.parquet
transformed_features.csv
run_config.json
```

`predictions.parquet` contains:

```text
epk_id, dataset, outcome, model, seed, evaluation_split, T, Y, cate_pred
```

A suite additionally produces `all_metrics.csv`, `summary.csv`, and `suite_config.json`. The summary contains the mean, standard deviation, and normal-approximation 95% half-width across seeds.

## Metrics

- normalized Qini AUC using the scikit-uplift convention;
- Qini area above the random-ranking line, scaled by `N²`;
- AUUC over the top-fraction outcome-rate difference curve;
- uplift@10% and uplift@20%;
- observed treated-minus-control mean difference (`ate_observed`) on the current evaluation split;
- mean and standard deviation of CATE predictions;
- fitting and prediction time.

Real marketing RCTs do not provide unit-level true CATE, so PEHE is not computed here. PEHE should be added only for semi-synthetic datasets such as IHDP or ACIC with known effects.

## Current Boundaries

1. This is a runnable and auditable first baseline version, not a line-by-line reproduction of every EconML/CATENets default.
2. T/X/DR use the same scikit-learn histogram gradient boosting base learner to control capacity fairly.
3. DR uses cross-fitting; T/X are evaluated on an independent test split.
4. The reported interval is currently across random seeds. Test-set bootstrap confidence intervals should still be added for the final protocol.
5. Orange Telecom is excluded because the treatment provenance and cleaned metadata remain causally ambiguous.
6. The cleaned Hillstrom data combine Men's and Women's email into `T=1`. This estimates “any marketing email vs no email,” not the paper's separate `Hill(1)` and `Hill(2)` tasks. Reconstruct two binary cohorts from the raw `segment` field for strict reproduction.

## Automatic Tuning

See [AUTO_TUNING_GUIDE.md](AUTO_TUNING_GUIDE.md) for full-data tuning, resume, and final-test commands. T/X/DR and DragonNet are tuned; pretrained CausalPFN is evaluated once with fixed parameters.


See [PARALLEL_TUNING_GUIDE.md](PARALLEL_TUNING_GUIDE.md) for detached multi-dataset scheduling and GPU assignment.
