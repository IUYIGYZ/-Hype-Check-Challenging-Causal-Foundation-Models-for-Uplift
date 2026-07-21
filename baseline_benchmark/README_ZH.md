# CausalPFN Uplift Baseline 对比框架

## 1. 目标

本目录提供一套可以直接运行的 baseline comparison，用于在相同的 cleaned 数据、相同样本划分、相同特征矩阵和相同指标下比较不同 CATE/Uplift 模型。

当前实现不包含 CausalPFN 本身；它先建立 baseline 端。后续接入 CausalPFN 时，只需要让它对同一测试集输出 `cate_pred`，再调用相同的 `evaluate_uplift` 即可。

## 2. 为什么选择这些方法

### T-Learner

分别在处理组和控制组拟合 `mu1(X)`、`mu0(X)`，预测差值作为 CATE。它是最直接的双模型基线。

### X-Learner

在 T-Learner 基础上构造 imputed treatment effects，再拟合 effect models。Künzel 等人的原论文指出它尤其适合处理组与控制组规模不平衡的情况，因此与 Criteo 的 85% treatment rate 和项目的小控制组鲁棒性实验直接相关。

### DR-Learner

使用 cross-fitted outcome nuisance predictions 构造 doubly robust pseudo-outcome，再拟合最终 CATE model。实现默认使用随机试验的经验 treatment rate 作为常数 propensity，并进行裁剪。它对应项目参考文献中讨论的 doubly robust score。

### DragonNet

使用共享 representation、两个 potential-outcome heads 和一个 propensity head，同时学习 Outcome 和 Treatment。当前实现是基本 DragonNet architecture，**没有加入原论文的 targeted regularization**；结果表中应明确这一点。如果需要与论文数字严格复现，应换用作者代码或 CATENets。

## 3. 文献依据

本实现选择与项目参考资料保持一致：

- 项目大纲：`../Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf`；
- CausalPFN：`../参考文献/2506.07918v2.pdf`；
- Q-Learner/低转化讨论：`../参考文献/2605.26288v1.pdf`；
- few-placebo、X/DR/GP-CATE：`../参考文献/2605.27473v1.pdf`。

外部原始/官方资料：

- Meta-learners 原论文（T/X）：https://doi.org/10.1073/pnas.1804597116
- DR-Learner 原论文：https://arxiv.org/abs/2004.14497
- DragonNet 原论文：https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN 官方仓库：https://github.com/vdblm/CausalPFN
- CATENets 官方仓库：https://github.com/AliciaCurth/CATENets
- scikit-uplift Qini 定义：https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html

## 4. 公平对比如何保证

`prepare_data` 只执行一次：

1. 从同一 cleaned Parquet 读取 `X/T/Y`；
2. 固定 primary Outcome；
3. 生成一次 train/validation/test split；
4. 只在训练集拟合 imputer、scaler 和 categorical encoder；
5. 为所有模型提供完全相同的转换后矩阵；
6. 调参时所有模型在同一 validation 上输出 `cate_pred`，参数冻结后再使用同一 test；
7. 所有预测调用同一套 Qini/AUUC 代码。

Criteo 和 LZD 使用完整特征向量 hash 作为 group，使重复特征向量不能跨越 train/validation/test。Hillstrom 和 RetailHero 按 `T × Y` 分层划分。

统一删除：

```text
epk_id
T（从 X 中移除，但作为单独 treatment 输入）
treatment_dt
split
```

Outcome 文件里的 `epk_id`、`lag` 和其他 Outcomes 不会进入 `X`。

## 5. 目录结构

```text
baseline_benchmark/
├── baseline_benchmark/
│   ├── data.py       # cleaned 数据读取、统一划分、统一预处理
│   ├── metrics.py    # Qini、AUUC、uplift@k
│   ├── models.py     # T/X/DR
│   └── neural.py     # DragonNet
├── tests/
│   └── test_smoke.py
├── run_baselines.py  # 单数据集、单 seed
├── run_suite.py      # 多数据集、多 seed，并生成汇总表
└── requirements.txt
```

## 6. 环境

当前默认 Conda base 环境可以运行传统 baseline，但其 PyTorch CUDA 动态库有冲突。机器上的 `Torch25` 环境已经验证可以运行所有传统和神经 baseline：

```bash
conda run -n Torch25 python -c "import torch; print(torch.__version__)"
```

推荐从本目录运行：

```bash
cd 项目/baseline_benchmark
```

`run_baselines.py` 和 `run_suite.py` 默认使用 `--evaluation-split validation`，用于调参和决定何时停止调参。参数冻结后，正式结果必须显式添加：

```text
--evaluation-split test
```

## 7. 快速验证

只运行传统模型：

```bash
python run_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner \
  --max-rows 5000 \
  --tree-max-iter 20 \
  --seed 17
```

运行神经模型：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models dragonnet \
  --max-rows 5000 \
  --epochs 20 \
  --seed 17
```

运行全部模型：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 50000 \
  --epochs 100 \
  --seed 0
```

`--max-rows 0` 表示使用全部 cleaned 数据。

## 8. 多随机种子验证与最终实验

### 调参停止建议

调参阶段只读取 validation 结果。使用 `qini_auc_normalized` 作为主指标，`uplift_at_10pct` 作为业务辅助指标。先用一个 seed 快速筛选，再让最好的 2～3 组设置使用至少 3 个 seeds 复核。

如果新设置的平均 validation Qini 提升小于 0.005，或提升小于 seed 波动带来的标准误，同时 Uplift@10% 没有明显改善，就可以停止调参。效果相近时选择更简单、运行更快的设置。参数冻结后才运行 test。

先用 2 个 seed 验证流程：

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1 \
  --max-rows 10000 \
  --epochs 20 \
  --tree-max-iter 50
```

使用 validation 比较参数；参数冻结后再对 test 运行 10 个 seed：

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --evaluation-split test \
  --max-rows 50000 \
  --epochs 100 \
  --tree-max-iter 150
```

不建议第一轮就对 Criteo 全量运行。Criteo conversion 很稀疏，50k 均匀子样本的控制组 conversion 事件可能很少。建议先完成其他三个数据集，再单独设计 Criteo 的样本规模和置信区间协议。

## 9. 输出文件

单次运行生成：

```text
results/<dataset>/<evaluation_split>_seed_<seed>_<timestamp>/
├── metrics.csv
├── predictions.parquet
├── splits.parquet
├── transformed_features.csv
└── run_config.json
```

其中 `predictions.parquet` 包含：

```text
epk_id, dataset, outcome, model, seed, evaluation_split, T, Y, cate_pred
```

多 seed suite 额外生成：

```text
all_metrics.csv
summary.csv
suite_config.json
```

`summary.csv` 给出每个指标的 mean、standard deviation 和基于 seeds 的 95% normal-approximation half-width。

## 10. 指标

当前统一报告：

- `qini_auc_normalized`：按照 scikit-uplift 定义归一化的 Qini AUC；
- `qini_coefficient`：实际 Qini 曲线相对随机直线的面积，并用 `N^2` 缩放；
- `auuc`：top-fraction uplift curve 的面积；
- `uplift_at_10pct`；
- `uplift_at_20pct`；
- 当前 evaluation split 的观测均值差 `ate_observed`；
- CATE 预测均值和标准差；
- 训练与推理时间。

真实营销 RCT 没有逐样本真实 CATE，因此这里不计算 PEHE。PEHE 只能在 IHDP/ACIC 等带真值的半合成数据上计算。

## 11. 当前实现的边界

1. 这是可运行、可审计的第一版 baseline，不宣称逐行复现 EconML/CATENets 的全部默认值。
2. T/X/DR 使用 scikit-learn HistGradientBoosting 作为统一底层学习器，便于公平控制模型容量。
3. DR-Learner 使用 cross-fitting；T/X 当前没有额外 cross-fitting，因为它们最终在独立测试集评估。
4. DragonNet 未实现 targeted regularization，必须在报告中写成 `DragonNet (basic, no targeted regularization)`。
5. 当前 CI 是跨随机种子的正态近似区间；项目大纲要求的 test-set bootstrap CI 仍应在最终实验阶段补充。
6. Orange Telecom 没有纳入可选数据集，因为其 treatment 来源和 cleaned metadata 仍存在因果有效性问题。
7. 当前 Hillstrom cleaned 数据把 Men's 与 Women's 两种邮件合并为 `T=1`。这可以解释为“任意营销邮件 vs 不发邮件”，但不等于 CausalPFN 论文中的 `Hill(1)` 和 `Hill(2)` 两个独立任务；若要严格复现，应从原始 `segment` 字段分别构造两个二元 cohort。

## 12. 正式 baseline 模型集合

本项目的 baseline 主表固定保留：

```text
T-Learner
X-Learner
DR-Learner
DragonNet (basic)
```

## 13. 自动调参

四个模型的完整自动调参、断点续跑和最终 test 命令见 [AUTO_TUNING_GUIDE.md](AUTO_TUNING_GUIDE.md)。


多数据集并行、GPU 分配和耐 SSH 断线运行见 [PARALLEL_TUNING_GUIDE.md](PARALLEL_TUNING_GUIDE.md)。
