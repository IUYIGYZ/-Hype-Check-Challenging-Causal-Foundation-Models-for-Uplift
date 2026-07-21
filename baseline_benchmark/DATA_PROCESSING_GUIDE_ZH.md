# Baseline Benchmark 数据处理说明

## 1. 这个文件是做什么的？

数据处理代码位于 `baseline_benchmark/data.py`。

它的任务不是训练模型，而是把已经清洗好的数据统一转换成所有 baseline 模型都能使用的格式。

整个流程是：

```text
cleaned Parquet 数据
        ↓
读取并对齐 X / T / Y
        ↓
排除 ID、干预时间等不能作为特征的列
        ↓
固定随机种子划分 train / validation / test
        ↓
只用训练集学习缺失值、标准化和 One-Hot 规则
        ↓
用同一套规则转换验证集和测试集
        ↓
生成 PreparedData
        ↓
所有 baseline 使用完全相同的数据进行对比
```

## 2. X、T、Y 分别是什么？

| 符号 | 名称 | 含义 | 例子 |
|---|---|---|---|
| X | 用户特征 | 干预之前已经知道的信息 | 历史消费、渠道、地区类型 |
| T | Treatment | 用户是否接受干预 | `T=1` 接受干预，`T=0` 未接受干预 |
| Y | Outcome | 希望干预改变的结果 | 是否转化、是否购买 |

本项目不是只预测 `Y`，而是估计同一个用户在“干预”和“不干预”两种情况下的结果差异：

```text
uplift(X) = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)
```

uplift 越大，说明这个用户越可能因为干预而转化。

## 3. 代码目前支持的数据集

| 代码名称 | cleaned 目录 | 默认 Y | 划分方式 |
|---|---|---|---|
| `criteo` | `Criteo-ITE-v2.1` | `conversion` | 重复特征向量分组划分 |
| `hillstrom` | `Hillstrom` | `conversion` | 按 T×Y 分层划分 |
| `lzd` | `LZD` | `Y` | 重复特征向量分组划分 |
| `retailhero` | `Retailhero-uplift` | `Y` | 按 T×Y 分层划分 |

Orange Telecom Churn 目前没有放入 baseline 的正式数据集配置，因为它的 Treatment 来源和数据版本还需要进一步确认。

### 3.1 Outcome 如何选择？

不同 Outcome 回答的问题不同，应当分开进行实验，不能在同一次实验中混用。

| Outcome | 类型 | 优点 | 局限 | 当前用法 |
|---|---|---|---|---|
| `conversion` | 二元 | 更接近最终购买这一商业目标 | 正样本少，Qini/AUUC 波动更大 | Criteo 和 Hillstrom 的 Primary Outcome |
| `visit` | 二元 | 事件数更多，小样本下通常更稳定 | 访问不等于最终购买 | Criteo/Hillstrom 可作为 Secondary Outcome |
| `spend` | 连续值 | 反映消费金额和经济价值 | 不是二元 Y | Hillstrom 中存在，当前 benchmark 暂不支持 |
| `Y` | 二元 | 数据集已经定义好的主结果 | 具体商业含义需以数据说明为准 | LZD 和 RetailHero 的 Primary Outcome |

Primary Outcome 是项目主表和主结论使用的结果；Secondary Outcome 是补充分析使用的结果。对 Hillstrom 和 Criteo，当前主实验使用 `conversion`：

```text
uplift(X) = P(conversion=1 | X,T=1) - P(conversion=1 | X,T=0)
```

`outcomes.parquet` 中的 `lag` 是技术/时间辅助字段，不是当前 Outcome。Criteo 的 `exposure` 与干预后是否实际曝光有关，可能是干预后变量，不应直接替换 `conversion` 作为主 Outcome。

## 4. cleaned 数据的文件结构

每个数据集至少需要两个文件：

```text
<dataset>/
├── features.parquet
└── outcomes.parquet
```

`features.parquet` 中包含：

- `epk_id`：样本 ID；
- `T`：干预标签；
- 其他干预前特征 X。

`outcomes.parquet` 中包含：

- `epk_id`：用于和特征表对齐；
- 一个或多个结果列，例如 `conversion`。

代码会检查两个文件的行数是否相同，并检查每一行的 `epk_id` 是否一致。如果 ID 顺序不一致，代码会停止，防止把某个用户的 X 和另一个用户的 Y 错误组合。

## 5. 数据读取和抽样

### 5.1 分批读取

Criteo 数据量很大。`_read_selected_rows()` 使用 PyArrow 分批读取 Parquet，默认每批最多读取 131,072 行，避免不必要地一次把整个文件加载到内存。

### 5.2 `max_rows`

`prepare_data()` 默认最多使用 50,000 行，便于快速调试。如果指定了较小的 `max_rows`，代码会使用固定 `seed` 无放回随机抽样。

同一个数据集、同一个 `max_rows` 和同一个 `seed` 会得到相同样本，从而保证模型对比的可复现性。
`max_rows` 可以用于整体数据规模的 scale stress test，但它不等于 CausalPFN 的 context size。`max_rows` 控制这次 benchmark 总共使用多少条观测数据；context size 控制 CausalPFN 在一次预测时看到多少个 context/support 样本。后者属于 CausalPFN 模型输入阶段，不由当前 `data.py` 直接实现。

注意：50,000 行只适合调试。Criteo 的 `conversion=1` 很稀少，正式实验需要使用更大样本或全量数据。

## 6. 哪些列不会进入模型？


代码统一排除：

```text
epk_id
T
treatment_dt
split
```

原因如下：

- `epk_id` 只是身份标识，不是可学习的用户特征；
- `T` 会作为单独的 Treatment 输入，不能重复混入 X；
- `treatment_dt` 是用户接受干预的时间，可能含有干预后信息，容易造成数据泄漏；
- `split` 是数据集划分标记，不是真实特征。

## 7. T 和 Y 的合法性检查

当前 benchmark 只支持二分类 Treatment 和二分类 Outcome。

代码要求：

```text
T 必须同时包含 0 和 1
Y 必须同时包含 0 和 1
```

如果只有处理组或只有控制组，就无法估计干预效果。如果 Y 全是 0 或全是 1，也没有可学习的结果差异。

## 8. 训练集、验证集和测试集如何划分？

默认比例是：

| 部分 | 默认比例 | 用途 |
|---|---:|---|
| Train | 60% | 训练模型，学习数据处理规则 |
| Validation | 20% | 模型选择、校准和调参 |
| Test | 20% | 最终公平评估 |

### 8.1 Hillstrom 和 RetailHero：分层划分

这两个数据集使用 `StratifiedShuffleSplit`，优先按 `T×Y` 的四种组合分层：

```text
T=0, Y=0
T=0, Y=1
T=1, Y=0
T=1, Y=1
```

目标是让 train、validation 和 test 中的处理组比例和正样本比例尽量相似。如果某个 T×Y 组合样本太少，代码会退化为只按 T 分层。

Hillstrom 中多个用户出现相同特征组合，主要是因为它的特征比较粗，例如地区类型、渠道和历史消费区间。这不等于同一个用户重复出现，因此不使用特征向量分组。

### 8.2 Criteo 和 LZD：重复特征向量分组划分

这两个数据集使用完整 X 的哈希值作为 group，再使用 `GroupShuffleSplit`。

两行数据如果具有完全相同的 X，它们一定会被放入同一个 split，不会一行在训练集、另一行在测试集。

这里的 group 是“完整特征向量组”，不是用户 ID。它的目的是降低重复样本造成的泄漏风险。

因为一个 group 不能被拆分，实际行数可能不会刚好是 60%/20%/20%，这是正常现象。

## 9. 划分完成后还会检查什么？

`_validate_splits()` 会检查：

1. train、validation、test 是否互不重叠；
2. 三部分是否完整覆盖所有样本；
3. 每个 split 是否都同时包含 `T=0` 和 `T=1`；
4. 每个 split 的每个 Treatment arm 中是否有足够的 `Y=1` 样本。

如果某个 arm 中的正样本少于 10，代码会发出：

```text
Qini/AUUC estimates will be unstable
```

这不一定代表代码错误，而是提醒当前样本太少，Qini/AUUC 可能有很大随机波动。正式实验应扩大数据量并使用多个随机种子。

## 10. 数值特征如何处理？

数值特征经过两个步骤。

### 10.1 中位数填补

如果数值列中存在缺失值，使用训练集该列的中位数填补。

例如：

```text
原始 age: 20, 25, NaN, 80
训练集中位数: 25
填补后: 20, 25, 25, 80
```

中位数相对不容易被极端值影响。

### 10.2 标准化

填补后使用 `StandardScaler`：

```text
x_scaled = (x - 训练集均值) / 训练集标准差
```

这会让不同数值特征具有比较接近的尺度，对神经网络等模型尤其重要。

## 11. 文本/类别特征如何变成数字？

代码把 pandas 类型为 `object`、`category` 或 `string` 的列判定为类别特征。

Hillstrom 中的典型类别特征包括：

```text
history_segment
zip_code
channel
```

处理顺序如下。

### 11.1 统一缺失值

先把 `None`、`pd.NA` 和 `NaN` 统一转换为可被 `SimpleImputer` 识别的缺失标记。

### 11.2 用众数填补

如果 `channel` 的训练数据是：

```text
Web, Phone, Web, <missing>
```

出现最多的是 `Web`，因此缺失值会被填成 `Web`。

### 11.3 One-Hot Encoding

模型不能直接把 `Web`、`Phone` 这类文字当数字计算，因此需要把每个类别拆成独立的 0/1 列。

例如：

| 原始 channel | channel_Phone | channel_Web | channel_Multichannel |
|---|---:|---:|---:|
| Phone | 1 | 0 | 0 |
| Web | 0 | 1 | 0 |
| Multichannel | 0 | 0 | 1 |

这些 0/1 不表示类别的大小或顺序，只表示某个样本是否属于该类别。

### 11.4 验证集/测试集出现新类别

`OneHotEncoder(handle_unknown="ignore")` 保证：如果测试集出现了训练集从未出现的类别，代码不会报错，对应的该组 One-Hot 列会全部记为 0。

## 12. 为什么处理规则只能在训练集上学习？

代码的执行方式是：

```python
preprocessor.fit_transform(X_train)
preprocessor.transform(X_val)
preprocessor.transform(X_test)
```

只有训练集使用 `fit_transform`。验证集和测试集只能使用训练集已经学到的：

- 数值列中位数；
- 数值列均值和标准差；
- 类别列众数；
- One-Hot 类别列表。

如果用全部数据学习这些规则，测试集的信息就会提前泄漏给训练流程，最终评估结果会偏乐观。

## 13. 模型最终收到什么？

`prepare_data()` 返回一个 `PreparedData` 对象：

| 字段 | 含义 |
|---|---|
| `X_train/X_val/X_test` | 已经完成填补、标准化和 One-Hot 的 `float32` 特征矩阵 |
| `t_train/t_val/t_test` | 干预标签 |
| `y_train/y_val/y_test` | 结果标签 |
| `id_train/id_val/id_test` | 用于追踪样本的 ID，不输入模型 |
| `feature_names` | 转换后的数值列和 One-Hot 列名 |
| `split_table` | 每个 `epk_id` 属于哪个 split |
| `preprocessor` | 已经在训练集拟合好的数据转换器 |
| `group_safe` | 当前数据集是否使用分组划分 |

转换完成后，代码还会检查特征矩阵中是否残留 `NaN` 或无穷大值。如果存在，会直接停止，避免错误数据进入模型。

## 14. 这样如何保证 baseline 对比公平？

在单次 baseline 实验中，`prepare_data()` 只执行一次。

所有模型收到相同的：

- 数据集；
- Outcome 定义；
- 抽样结果；
- train/validation/test 样本；
- 处理后特征矩阵；
- T 和 Y；
- 评估指标。

只有模型的学习方法不同。因此 T-Learner、X-Learner、DR-Learner 和 DragonNet 之间的结果才具有可比性。

## 15. 如何直接调用？

```python
from pathlib import Path
from baseline_benchmark.data import prepare_data

data = prepare_data(
    cleaned_root=Path("../data/data_A_cleaned"),
    dataset="hillstrom",
    outcome="conversion",
    max_rows=50_000,
    seed=42,
    val_fraction=0.2,
    test_fraction=0.2,
)

print(data.X_train.shape)
print(data.feature_names)
```

如果通过 baseline 入口运行：

```bash
cd 项目/baseline_benchmark

conda run -n Torch25 python run_baselines.py \
  --dataset hillstrom \
  --models t_learner,x_learner,dr_learner,dragonnet \
  --max-rows 50000 \
  --seed 42
```

## 16. 当前状态

数据处理代码已经通过自动测试，并对 Hillstrom、RetailHero、LZD 和 Criteo 分别完成了 5,000 行小规模干跑。四个数据集都能产生维度一致且不含 `NaN/Inf` 的 train、validation 和 test 特征矩阵。

当前数据处理流程可以用于第一轮 baseline 模型对比。正式结果阶段还需要：

1. 对稀疏 Outcome 数据集使用更大样本；
2. 使用多个随机种子；
3. 报告指标的均值、标准差或置信区间；
4. 保存并复用相同的 split 和 preprocessor。
