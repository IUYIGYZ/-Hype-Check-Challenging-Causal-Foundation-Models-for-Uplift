# Baseline 自动调参运行指南（中文） （The English version is under the Chinese version）

## 1. 这个脚本做什么

`tune_baselines.py` 对以下四个模型分别执行可复现的随机搜索：

- T-Learner
- X-Learner
- DR-Learner
- DragonNet (basic, no targeted regularization)

数据只准备一次。所有 trial 共享完全相同的抽样结果、train/validation/test 划分、预处理器和转换后特征矩阵。每个模型独立选择自己的超参数。

默认优化指标是 validation 集上的 `qini_auc_normalized`。若该指标相同，则依次比较 validation `uplift_at_10pct` 和训练时间。调参期间不会查看 test；只有显式加入 `--final-test` 才会在选参结束后评估 test。

## 2. “使用全部数据”的准确含义

`--max-rows 0` 表示读取该 cleaned 数据集的全部行，不进行行抽样。这仍然会把全部行划分成 train、validation 和 test：

- train：拟合模型；
- validation：选择超参数，并供 DragonNet early stopping 使用；
- test：调参期间保持封存。

因此，“完整数据调参”不等于把 test 也拿去训练。这样才能得到公平、可信的最终结果。

## 3. 搜索空间

| 模型 | 自动搜索参数 |
|---|---|
| T-Learner | max_iter、max_leaf_nodes、learning_rate |
| X-Learner | max_iter、max_leaf_nodes、learning_rate |
| DR-Learner | max_iter、max_leaf_nodes、learning_rate、n_folds |
| DragonNet | batch_size、hidden_dim、learning_rate、weight_decay、patience |

DragonNet 的最大 epoch 由 `--neural-max-epochs` 控制，validation early stopping 会在长期没有改善时提前停止。每个模型的 trial 0 都是当前默认参数，后续 trial 才是随机搜索参数。

## 4. 小样本试运行

先进入项目目录：

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark
```

用 5,000 行、每个模型 2 个 trial、DragonNet 最多 5 个 epoch 检查流程：

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 5000 \
  --trials 2 \
  --neural-max-epochs 5 \
  --device cpu
```

这只是程序测试，不能作为最终模型结果。

## 5. 使用完整数据正式调参

下面命令不抽样，四个模型各运行 20 个 trial：

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto
```

第一次正式调参不要加入 `--final-test`。先查看 `best_params.json` 和 `trial_metrics.csv`，确认所有模型都有成功 trial，再冻结参数。

要调其他数据集，把 `retailhero` 分别换成：

```text
hillstrom
lzd
criteo
```

每个数据集必须单独运行和保存结果。Criteo 全量约有千万级样本，20×4 个 trial 的时间和内存成本可能很高；可以先用少量 trial 验证全量资源需求，再增加 trial 数。

## 6. 中断后继续

每个 trial 结束后，脚本会立即保存 `trial_metrics.csv` 和 `best_params.json`。找到原运行目录后，用相同数据和搜索设置继续：

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto \
  --resume-dir tuning_results/retailhero/tuning_seed_42_YYYYMMDD_HHMMSS
```

已完成的 trial 会显示 `SKIP completed`。恢复时必须保持 dataset、outcome、模型列表、数据路径、数据划分 seed、search seed、划分比例、目标指标、device 和 epoch 设置一致。trial 数可以增加，例如从 20 增加到 30，脚本只补做新增 trial。

## 7. 最终只评估一次 test

参数检查并冻结后，在原调参目录上增加 `--final-test`：

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto \
  --resume-dir tuning_results/retailhero/tuning_seed_42_YYYYMMDD_HHMMSS \
  --final-test
```

如果该目录已经存在 `final_test_metrics.csv`，脚本会拒绝再次评估，避免反复看 test 后继续调参造成 test leakage。

## 8. 输出文件

```text
tuning_results/<dataset>/tuning_seed_<seed>_<timestamp>/
├── tuning_config.json
├── data_manifest.json
├── splits.parquet
├── transformed_features.csv
├── preprocessor.joblib
├── trial_metrics.csv
├── best_params.json
├── final_test_metrics.csv                 # 仅 --final-test
└── final_test_predictions/                # 仅 --final-test
    ├── t_learner.parquet
    ├── x_learner.parquet
    ├── dr_learner.parquet
    └── dragonnet.parquet
```

`trial_metrics.csv` 是完整调参记录；`best_params.json` 是每个模型当前最佳 validation 配置；`data_manifest.json` 用于确认是否使用全量数据以及三个集合的样本、干预和 outcome 分布。

---

# Baseline Automatic Tuning Guide (English)

## 1. What the script does

`tune_baselines.py` performs reproducible random search independently for T-Learner, X-Learner, DR-Learner, and DragonNet (basic, without targeted regularization).

Data are prepared once. Every trial shares exactly the same sampled rows, train/validation/test split, fitted preprocessor, and transformed feature matrices. Each model selects its own hyperparameters.

The default objective is validation `qini_auc_normalized`. Ties are broken by validation `uplift_at_10pct`, then by shorter fitting time. Test data are never evaluated during tuning. They are evaluated only when `--final-test` is explicitly supplied.

## 2. Meaning of full-data tuning

`--max-rows 0` reads every row in the cleaned dataset and disables row subsampling. Those rows are still separated into train, validation, and test:

- train fits the model;
- validation selects hyperparameters and supports DragonNet early stopping;
- test remains sealed during tuning.

Full-data tuning therefore does not train on test data.

## 3. Search space

| Model | Tuned parameters |
|---|---|
| T-Learner | max_iter, max_leaf_nodes, learning_rate |
| X-Learner | max_iter, max_leaf_nodes, learning_rate |
| DR-Learner | max_iter, max_leaf_nodes, learning_rate, n_folds |
| DragonNet | batch_size, hidden_dim, learning_rate, weight_decay, patience |

DragonNet's maximum epochs are controlled by `--neural-max-epochs`; validation early stopping may stop earlier. Trial 0 uses the current default configuration, and later trials sample the search space.

## 4. Small dry run

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark

conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 5000 \
  --trials 2 \
  --neural-max-epochs 5 \
  --device cpu
```

This command checks the pipeline only. Its metrics are not final experimental results.

## 5. Formal tuning with every cleaned row

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto
```

Do not add `--final-test` to the initial tuning run. Inspect `best_params.json` and `trial_metrics.csv`, confirm that every model has successful trials, and freeze the settings first.

Run the command separately for `retailhero`, `hillstrom`, `lzd`, and `criteo`. Full Criteo tuning can be very expensive because it has millions of rows. A low-trial full-data resource check is sensible before increasing the trial count.

## 6. Resume an interrupted run

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto \
  --resume-dir tuning_results/retailhero/tuning_seed_42_YYYYMMDD_HHMMSS
```

Completed trials are skipped. Dataset, outcome, model list, data path, split seed, search seed, split fractions, objective, device, and epoch settings must match the original run. Trial counts may be increased; only the new trial indices are executed.

## 7. Evaluate test exactly once

After reviewing and freezing the selected settings, rerun the resume command with `--final-test`:

```bash
conda run -n Torch25 python tune_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --device auto \
  --resume-dir tuning_results/retailhero/tuning_seed_42_YYYYMMDD_HHMMSS \
  --final-test
```

If `final_test_metrics.csv` already exists, the script refuses another final evaluation to reduce test leakage.

## 8. Outputs

Each run stores its configuration, data manifest, exact split, fitted preprocessor, transformed feature names, all trial metrics, and best validation parameters. The optional final-test run additionally stores test metrics and one prediction parquet file per model.

## 9. Parallel and detached execution

See [PARALLEL_TUNING_GUIDE.md](PARALLEL_TUNING_GUIDE.md) for multi-dataset GPU scheduling that survives SSH logout.
