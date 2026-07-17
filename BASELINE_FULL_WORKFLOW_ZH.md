# CausalPFN 项目 Baseline 完整工作说明（中文版）

## 1. 范围与目标

本文档说明当前已经完成的 baseline 部分：cleaned 数据整理、统一数据划分与编码、传统和神经 CATE 模型、Uplift 指标、批量实验、输出文件、代码审查结果和已知限制。

实现目录：`项目/baseline_benchmark/`。本阶段尚未接入 CausalPFN；后续只需让 CausalPFN 对相同测试 ID 输出 `cate_pred`，即可使用同一评估代码公平比较。

任务输入为干预前特征 `X`、二元处理 `T` 和二元结果 `Y`，模型估计：

\[
\tau(x)=E[Y(1)-Y(0)\mid X=x].
\]

真实营销 RCT 没有逐样本真实 CATE，因此主要评价排序质量，而不是普通分类准确率。

## 2. 数据集整理

Baseline 从 `项目/data/data_A_cleaned/` 读取数据。

| 数据集 | 行数 | 默认 Outcome | 数据处理状态 | 使用建议 |
|---|---:|---|---|---|
| Criteo | 13,979,592 | `conversion` | 删除冗余特征；重复特征向量需 group-safe split | 可用，但 conversion 极稀疏 |
| Hillstrom | 64,000 | `conversion` | 保留文本类别，运行时 one-hot | 可用，低转化 |
| LZD | 181,669 | `Y` | 仅保留随机试验 test 部分；删除冗余特征 | 可用 |
| RetailHero | 200,039 | `Y` | 仅保留有标签部分；删除疑似干预后特征 | 可用 |
| Orange Telecom | 11,896 | `churn` | Treatment 来源和 metadata 有矛盾 | 暂不纳入主实验 |

当前 Hillstrom 把 Men's 和 Women's email 合并为 `T=1`，因此估计的是“任意邮件 vs 不发邮件”。严格复现 CausalPFN 论文的 `Hill(1)`/`Hill(2)` 时，应从原始 `segment` 分别建立两个二元 cohort。

## 3. 数据处理流程

### 3.1 读取与校验

`baseline_benchmark/data.py`：

1. 读取 `features.parquet` 和 `outcomes.parquet`；
2. 检查行数一致、`epk_id` 逐行对齐；
3. 检查 `T` 同时包含 0/1，选定 Outcome 为二元 0/1；
4. 大数据通过 PyArrow 分批读取选中行；
5. 最终检查数值矩阵无 NaN/Inf。

默认 Outcome：

```yaml
criteo: conversion
hillstrom: conversion
lzd: Y
retailhero: Y
```

统一不进入模型特征：

```text
epk_id, T, treatment_dt, split, lag, 未选中的其他 Outcomes
```

`T` 作为独立 Treatment 数组保留。Criteo 的 `exposure` 是干预后变量，不进入 `X`，也不作为主 Outcome。

### 3.2 子样本

`--max-rows N` 按固定 seed 无放回抽样；`--max-rows 0` 使用全部数据。同一次运行的所有模型共享相同样本。

Criteo conversion 极低，50k 均匀样本中的控制组正事件可能很少；正式 Criteo 实验应扩大样本并报告 bootstrap CI。

### 3.3 Train/Validation/Test

默认比例：

```text
train 60% / validation 20% / test 20%
```

- Hillstrom、RetailHero：按 `T × Y` 使用 `StratifiedShuffleSplit`；
- Criteo、LZD：按完整特征向量 hash 使用 `GroupShuffleSplit`，防止重复特征向量跨 split 泄漏。

划分后检查：split 互斥且完整覆盖；每个 split 同时包含两个 Treatment arm；每个 arm 少于 10 个正事件时发出不稳定性警告。

### 3.4 文本、缺失值和数值处理

类别列自动识别为 `object/category/string`：

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

Hillstrom 的 `history_segment`、`zip_code`、`channel` 均转换为 one-hot 数值列。例如 `zip_code=Urban` 转换为 `zip_code_Urban=1`。

数值列：

```python
SimpleImputer(strategy="median")
StandardScaler()
```

正确顺序是先划分，只在 `X_train` 上拟合预处理器，再 transform train/validation/test，防止测试信息进入缺失值统计、标准化参数或类别词表。

所有模型共享同一组最终数组：

```text
X_train/T_train/Y_train
X_validation/T_validation/Y_validation
X_test/T_test/Y_test
```

## 4. Baseline 模型

传统方法统一使用 scikit-learn `HistGradientBoostingClassifier/Regressor`，尽量控制底层学习器差异。

### Constant ATE

所有测试样本输出训练集 `mean(Y|T=1)-mean(Y|T=0)`；无排序能力，用作指标 sanity check，normalized Qini/Uplift AUC 应为 0。

### S-Learner

拟合一个 `mu(X,T)`：

\[
\hat\tau(x)=\hat\mu(x,1)-\hat\mu(x,0).
\]

### T-Learner

分别拟合 `mu0(X)` 和 `mu1(X)`：

\[
\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x).
\]

### X-Learner

构造 `D0=mu1(X0)-Y0`、`D1=Y1-mu0(X1)`，拟合 `tau0/tau1`，使用经验 propensity `g=P(T=1)`：

\[
\hat\tau(x)=g\hat\tau_0(x)+(1-g)\hat\tau_1(x).
\]

它是处理组规模不平衡时的重要基线。

### DR-Learner

通过分层 K-fold cross-fitting 获得 out-of-fold `mu0/mu1`，在 RCT 中使用经验常数 propensity `e`：

\[
\tilde\tau=\hat\mu_1-\hat\mu_0+
\frac{T(Y-\hat\mu_1)}{e}-
\frac{(1-T)(Y-\hat\mu_0)}{1-e}.
\]

最终回归 `E[tilde_tau|X]`。每个 `T × Y` stratum 少于两个样本时明确报错。

### TARNet

PyTorch 共享 representation 加 control/treated 两个 outcome heads，只对实际观察到的 head 计算 BCE，并以 validation loss early stopping。

### DragonNet

在 TARNet 上增加 propensity head，损失为 outcome BCE + Treatment BCE。当前是 `DragonNet (basic, no targeted regularization)`，报告必须明确；严格复现应使用作者实现或 CATENets。

神经代码设置 NumPy/PyTorch seed、deterministic algorithms、cuDNN deterministic 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。本机神经实验使用已验证的 `Torch25` Conda 环境。

## 5. 指标

指标位于 `baseline_benchmark/metrics.py`，所有模型统一调用。

- `qini_auc_normalized`：与 scikit-uplift `qini_auc_score` 兼容；
- `qini_coefficient`：实际 Qini 曲线超过随机直线的面积，除以 `N²`；
- `uplift_auc_normalized`：与 scikit-uplift `uplift_auc_score` 兼容；
- `auuc`：标准 Uplift gain curve 原始面积，除以 `N²`；
- `uplift_at_10pct`、`uplift_at_20pct`：top-k 中处理组与控制组结果率差；
- `ate_test`、CATE 均值/标准差、fit/predict 时间。

相同预测分数会作为同一 threshold 处理，避免 tie 内部行顺序影响 Qini/Uplift 曲线面积。top-k 缺少任一 Treatment arm 时返回 NaN。

真实数据没有真实 `tau(x)`，因此不计算 PEHE；IHDP/ACIC 接入后再增加。

## 6. 代码与运行入口

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

进入目录：

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark
```

传统模型快速验证：

```bash
python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner \
  --max-rows 5000 --tree-max-iter 20 --seed 17
```

神经模型：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero --models tarnet,dragonnet \
  --max-rows 5000 --epochs 20 --seed 17
```

全部 baseline：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet \
  --max-rows 50000 --epochs 100 --tree-max-iter 150 --seed 0
```

添加 `--save-transformed-data` 可导出最终数值 train/validation/test Parquet，供后续独立运行的 CausalPFN 复用。

多数据集、10 seeds：

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-rows 50000 --epochs 100 --tree-max-iter 150
```

## 7. 输出与整理

单次运行：

```text
results/<dataset>/seed_<seed>_<timestamp>/
├── data_manifest.json
├── metrics.csv
├── predictions.parquet
├── preprocessor.joblib
├── run_config.json
├── splits.parquet
├── transformed_features.csv
└── prepared_data/             # 使用 --save-transformed-data 时
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

- `data_manifest.json`：路径、Outcome、seed、group-safe 状态、特征名、各 split 的样本/arm/事件统计；
- `preprocessor.joblib`：训练集拟合的 imputer/scaler/one-hot 和列顺序；
- `splits.parquet`：`epk_id, split`；
- `predictions.parquet`：`epk_id,dataset,outcome,model,seed,T,Y,cate_pred`；
- `metrics.csv`：每个模型的指标与耗时；
- `prepared_data/*.parquet`：最终数值矩阵和 `epk_id/T/Y`。

`run_suite.py` 另外生成 `all_metrics.csv`、`summary.csv`、`suite_config.json`。`summary.csv` 报告跨 seeds 的 mean、standard deviation 和 95% normal-approximation half-width。

## 8. 公平实验保证

1. 同一抽样样本和 split；
2. 相同转换后 `X/T/Y`；
3. 相同测试 ID；
4. 相同指标函数；
5. 预处理仅在训练集拟合；
6. split、预处理器、配置、预测和指标全部保存。

当前是固定超参数第一版，尚未完成正式调参。正式主表应在 validation 上为各模型定义预算相近的搜索协议，test 仅用于最终评估。

## 9. 代码审查结果

本轮检查覆盖数据读取、ID 对齐、分层/group-safe split、未知类别、train-only preprocessing、S/T/X/DR 公式、DR cross-fitting、神经训练、Qini/Uplift 官方定义、结果输出和 CUDA 可复现性。

修复内容：

1. 增加标准 `uplift_auc_normalized`，消除原 `auuc` 命名歧义；
2. 增加 split 完整性、Treatment arm 和低事件数检查；
3. 增加 DR 最小 stratum 检查；
4. 增加 `preprocessor.joblib`、`data_manifest.json` 和可选数值矩阵导出；
5. 增加 CUDA/cuBLAS deterministic 配置；
6. 修复 NumPy 1.23 分层字符串兼容和静态格式问题。

最终验证：

```text
pytest: 5 passed
flake8: 0 errors
mypy: 0 errors
syntax compile: passed
traditional end-to-end: passed
TARNet/DragonNet in Torch25: passed
prepared-data export: passed
```

pytest 的两条 warning 来自环境 numexpr 使用旧 `distutils`，不是本项目代码错误。烟雾测试只证明运行正确，不应当作科学结果。

## 10. 当前限制与下一步

- 尚未接入 CausalPFN；下一步应复用同一 split/数值矩阵和指标。
- 尚未包含 Causal Forest，因为当前环境没有 EconML；最终主表建议增加 CausalForestDML/GRF。
- DragonNet 未包含 targeted regularization。
- 尚未在 validation 上完成正式超参数搜索。
- 当前 CI 是跨 seed 近似区间，最终仍需 test-set bootstrap 95% CI。
- Criteo 需要更大样本和单独的低转化协议。
- Q-Learner、GP-CATE 属于后续 robustness 专项；CausalPFN-Rank 应在找到默认模型失效场景后实现。

## 11. 参考资料

项目内：`Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf`、`参考文献/2506.07918v2.pdf`、`2605.26288v1.pdf`、`2605.27473v1.pdf`。

外部原论文/官方资料：

- S/T/X：https://doi.org/10.1073/pnas.1804597116
- DR-Learner：https://arxiv.org/abs/2004.14497
- TARNet/CFR：https://proceedings.mlr.press/v70/shalit17a.html
- DragonNet：https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN：https://github.com/vdblm/CausalPFN
- EconML：https://econml.azurewebsites.net/
- CATENets：https://github.com/AliciaCurth/CATENets
- scikit-uplift 指标：https://www.uplift-modeling.com/en/latest/_modules/sklift/metrics/metrics.html
