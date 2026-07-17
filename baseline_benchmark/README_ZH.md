# CausalPFN Uplift Baseline 对比框架

## 1. 目标

本目录提供一套可以直接运行的 baseline comparison，用于在相同的 cleaned 数据、相同样本划分、相同特征矩阵和相同指标下比较不同 CATE/Uplift 模型。

当前实现不包含 CausalPFN 本身；它先建立 baseline 端。后续接入 CausalPFN 时，只需要让它对同一测试集输出 `cate_pred`，再调用相同的 `evaluate_uplift` 即可。

## 2. 为什么选择这些方法

### Constant ATE

为所有用户输出相同的训练集平均处理效应。它没有个体排序能力，是检查 Qini 实现是否正常的 sanity check，而不是主要竞争模型。正常情况下其 Qini coefficient 应接近 0。

### S-Learner

使用单个 outcome model 学习 `E[Y | X,T]`，然后分别令 `T=1` 和 `T=0` 得到 CATE。它简单、训练成本低，也是项目大纲提到的重要基线。

### T-Learner

分别在处理组和控制组拟合 `mu1(X)`、`mu0(X)`，预测差值作为 CATE。它是最直接的双模型基线。

### X-Learner

在 T-Learner 基础上构造 imputed treatment effects，再拟合 effect models。Künzel 等人的原论文指出它尤其适合处理组与控制组规模不平衡的情况，因此与 Criteo 的 85% treatment rate 和项目的小控制组鲁棒性实验直接相关。

### DR-Learner

使用 cross-fitted outcome nuisance predictions 构造 doubly robust pseudo-outcome，再拟合最终 CATE model。实现默认使用随机试验的经验 treatment rate 作为常数 propensity，并进行裁剪。它对应项目参考文献中讨论的 doubly robust score。

### TARNet

PyTorch 神经基线。使用共享 representation 和两个 potential-outcome heads，只在每个样本实际观察到的 treatment head 上计算二元交叉熵。

### DragonNet

在 TARNet 结构上增加 propensity head，同时学习 Outcome 和 Treatment。当前实现是基本 DragonNet architecture，**没有加入原论文的 targeted regularization**；结果表中应明确这一点。如果需要与论文数字严格复现，应换用作者代码或 CATENets。

## 3. 文献依据

本实现选择与项目参考资料保持一致：

- 项目大纲：`../Hype_Check__Challenging_Causal_Foundation_Models_for_Uplift.pdf`；
- CausalPFN：`../参考文献/2506.07918v2.pdf`；
- Q-Learner/低转化讨论：`../参考文献/2605.26288v1.pdf`；
- few-placebo、X/DR/GP-CATE：`../参考文献/2605.27473v1.pdf`。

外部原始/官方资料：

- Meta-learners 原论文（S/T/X）：https://doi.org/10.1073/pnas.1804597116
- DR-Learner 原论文：https://arxiv.org/abs/2004.14497
- TARNet/CFR 原论文：https://proceedings.mlr.press/v70/shalit17a.html
- DragonNet 原论文：https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects
- CausalPFN 官方仓库：https://github.com/vdblm/CausalPFN
- EconML 官方文档：https://econml.azurewebsites.net/
- CATENets 官方仓库：https://github.com/AliciaCurth/CATENets
- scikit-uplift Qini 定义：https://www.uplift-modeling.com/en/stable/api/metrics/qini_auc_score.html

## 4. 公平对比如何保证

`prepare_data` 只执行一次：

1. 从同一 cleaned Parquet 读取 `X/T/Y`；
2. 固定 primary Outcome；
3. 生成一次 train/validation/test split；
4. 只在训练集拟合 imputer、scaler 和 categorical encoder；
5. 为所有模型提供完全相同的转换后矩阵；
6. 所有模型在同一个测试集上输出 `cate_pred`；
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
│   ├── models.py     # Constant/S/T/X/DR
│   └── neural.py     # TARNet/DragonNet
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

## 7. 快速验证

只运行传统模型：

```bash
python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner \
  --max-rows 5000 \
  --tree-max-iter 20 \
  --seed 17
```

运行神经模型：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models tarnet,dragonnet \
  --max-rows 5000 \
  --epochs 20 \
  --seed 17
```

运行全部模型：

```bash
conda run -n Torch25 python run_baselines.py \
  --dataset retailhero \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner,tarnet,dragonnet \
  --max-rows 50000 \
  --epochs 100 \
  --seed 0
```

`--max-rows 0` 表示使用全部 cleaned 数据。

## 8. 正式多随机种子实验

先用 2 个 seed 验证流程：

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1 \
  --max-rows 10000 \
  --epochs 20 \
  --tree-max-iter 50
```

确认没有问题后再运行 10 个 seed：

```bash
conda run -n Torch25 python run_suite.py \
  --datasets retailhero,lzd,hillstrom \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-rows 50000 \
  --epochs 100 \
  --tree-max-iter 150
```

不建议第一轮就对 Criteo 全量运行。Criteo conversion 很稀疏，50k 均匀子样本的控制组 conversion 事件可能很少。建议先完成其他三个数据集，再单独设计 Criteo 的样本规模和置信区间协议。

## 9. 输出文件

单次运行生成：

```text
results/<dataset>/seed_<seed>_<timestamp>/
├── metrics.csv
├── predictions.parquet
├── splits.parquet
├── transformed_features.csv
└── run_config.json
```

其中 `predictions.parquet` 包含：

```text
epk_id, dataset, outcome, model, seed, T, Y, cate_pred
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
- 测试集原始 `ATE`；
- CATE 预测均值和标准差；
- 训练与推理时间。

真实营销 RCT 没有逐样本真实 CATE，因此这里不计算 PEHE。PEHE 只能在 IHDP/ACIC 等带真值的半合成数据上计算。

## 11. 当前实现的边界

1. 这是可运行、可审计的第一版 baseline，不宣称逐行复现 EconML/CATENets 的全部默认值。
2. S/T/X/DR 使用 scikit-learn HistGradientBoosting 作为统一底层学习器，便于公平控制模型容量。
3. DR-Learner 使用 cross-fitting；S/T/X 当前没有额外 cross-fitting，因为它们最终在独立测试集评估。
4. DragonNet 未实现 targeted regularization，必须在报告中写成 `DragonNet (basic, no targeted regularization)`。
5. 当前没有 Causal Forest，因为环境未安装 EconML。正式论文级实验建议后续安装 EconML，并增加 `CausalForestDML` 或 GRF。
6. 当前 CI 是跨随机种子的正态近似区间；项目大纲要求的 test-set bootstrap CI 仍应在最终实验阶段补充。
7. Orange Telecom 没有纳入可选数据集，因为其 treatment 来源和 cleaned metadata 仍存在因果有效性问题。
8. 当前 Hillstrom cleaned 数据把 Men's 与 Women's 两种邮件合并为 `T=1`。这可以解释为“任意营销邮件 vs 不发邮件”，但不等于 CausalPFN 论文中的 `Hill(1)` 和 `Hill(2)` 两个独立任务；若要严格复现，应从原始 `segment` 字段分别构造两个二元 cohort。

## 12. 推荐的正式模型集合

如果计算预算有限，主表至少保留：

```text
Constant ATE
S-Learner
T-Learner
X-Learner
DR-Learner
TARNet
DragonNet (basic)
CausalPFN（下一步接入）
```

若时间与环境允许，再增加：

```text
Causal Forest / GRF
CATENets 官方 TARNet/DragonNet
Q-Learner（低转化专项）
GP-CATE（small-control 专项）
```

