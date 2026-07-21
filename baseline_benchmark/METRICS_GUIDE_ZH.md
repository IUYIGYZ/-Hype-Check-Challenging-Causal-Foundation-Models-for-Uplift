# `metrics.py` 代码与评估指标说明

## 1. 文件作用

`baseline_benchmark/metrics.py` 负责评估 uplift/CATE 模型的效果。

它不读取数据、不处理特征、不训练模型，只接收模型已经生成的 CATE 预测分数，然后评估模型能否把“更可能因为干预而购买的用户”排在前面。

它在整个 baseline 流程中的位置是：

```text
处理后的 X_train / X_val / X_test
                  ↓
       模型训练并预测 CATE
                  ↓
     metrics.py 评估 CATE 排序
                  ↓
 Qini、AUUC、Uplift@10%、Uplift@20%
```

---

## 2. 主要输入

主入口是：

```python
evaluate_uplift(y, score, treatment)
```

三个输入必须一一对应：

| 输入 | 含义 | 当前取值 |
|---|---|---|
| `y` | 用户最终结果，例如是否购买 | `0` 或 `1` |
| `treatment` | 用户是否接受干预 | `0` 或 `1` |
| `score` | 模型预测的 CATE/uplift | 有限实数 |

其中：

```text
T = 1：接受干预
T = 0：未接受干预，即 control
Y = 1：发生目标事件，例如购买
Y = 0：没有发生目标事件
```

`score` 表示模型预测的条件平均干预效果：

\[
\hat\tau(x)
=
\widehat P(Y=1\mid X=x,T=1)
-
\widehat P(Y=1\mid X=x,T=0)
\]

例如，某个用户的 `score = 0.08` 表示：

> 模型预测干预会使该类用户的购买概率提高约 0.08，即 8 个百分点。

这是模型预测，不是可直接观测到的个体真实效果。

---

## 3. 为什么不使用普通准确率作为主指标

原因有两个。

### 3.1 商业目标不同

普通购买预测回答：

> 这个用户最后会不会购买？

Uplift 模型回答：

> 这个用户是否会因为干预而提高购买概率？

一个本来就会购买的用户，可能很容易被普通分类器正确预测，但并不一定值得向他发送优惠券。

### 3.2 个体反事实不可同时观测

同一个用户只能出现一种状态：

- 接受干预后的结果；
- 或者没有接受干预时的结果。

我们无法同时观察同一个用户的两个潜在结果，因此个体真实 CATE 通常不可见，不能直接计算普通的 CATE 预测误差。

所以本文件通过 treatment/control 两组的统计差异，评估模型的 uplift 排序能力。

---

## 4. `_as_arrays`：输入检查

`_as_arrays` 会将三个输入转换成一维 NumPy 数组，并检查：

1. `y`、`score`、`treatment` 长度必须相同；
2. 数组不能为空；
3. 不能包含 `NaN` 或无穷大；
4. `y` 必须是二元结果；
5. `treatment` 必须使用 `0/1` 编码；
6. 评估集中必须同时存在 treatment 和 control。

如果评估集只有一个 treatment arm，就无法计算两组结果差，因此代码会直接报错。

---

## 5. `_threshold_indices`：处理相同的预测分数

模型可能给多个用户相同的 CATE，例如：

```text
0.20, 0.20, 0.20, 0.10, 0.10
```

`_threshold_indices` 仅在分数发生变化的位置计算曲线，因此相同分数会作为一个整体进入 Qini/Uplift 曲线。

这可以避免曲线被并列用户的任意行顺序影响。

---

## 6. `qini_curve`：Qini 曲线

代码首先按预测 CATE 从高到低排序用户。

对排名前 \(k\) 个用户，定义：

- \(N_t(k)\)：treatment 用户数量；
- \(N_c(k)\)：control 用户数量；
- \(Y_t(k)\)：treatment 中发生目标事件的人数；
- \(Y_c(k)\)：control 中发生目标事件的人数。

Qini gain 计算为：

\[
Q(k)
=
Y_t(k)
-
Y_c(k)\frac{N_t(k)}{N_c(k)}
\]

它表示：

> 排名前 \(k\) 个用户中，treatment 组实际事件数，减去根据 control 组推算的无干预预期事件数。

例如：

```text
排名前 1000 人：
treatment：500 人，50 人购买
control：  500 人，30 人购买
```

则：

\[
Q(1000)=50-30\times\frac{500}{500}=20
\]

可以理解为，当前选中人群中估计额外发生了约 20 个购买事件。

当某个早期排名位置还没有 control 用户时，无法做除法，代码暂时将该位置的 gain 设为 0。

---

## 7. `qini_auc_normalized`：归一化 Qini AUC

该指标比较：

- 当前模型的 Qini 曲线；
- 随机排序基准线；
- 根据观测的 `Y` 和 `T` 构造的理想 Qini 曲线。

计算方式是：

\[
\text{Normalized Qini}
=
\frac{
AUC_{model}-AUC_{random}
}{
AUC_{perfect}-AUC_{random}
}
\]

基本解读：

- 大于 0：整体排序优于随机；
- 等于 0：与随机排序或常数预测相当；
- 小于 0：排序较差，甚至可能排反；
- 越大通常越好。

本项目的自动调参默认使用 `qini_auc_normalized` 作为 validation objective。

### 取值注意事项

`qini_auc_normalized` 不是概率，不能简单假设它严格位于 `[0, 1]`。

在样本很小、正例很少或 treatment/control 不平衡时，经验曲线波动可能导致它大于 1 或小于 -1。这不一定是代码错误，但通常意味着需要检查样本量和结果稳定性。

---

## 8. `qini_coefficient`：未归一化 Qini 面积

计算方式为：

\[
\frac{
AUC_{model}-AUC_{random}
}{N^2}
\]

除以 \(N^2\) 是为了减少数据集样本数对数值量级的直接影响。

它与 `qini_auc_normalized` 的区别是：

- `qini_auc_normalized` 还会除以理想曲线超过随机线的面积；
- `qini_coefficient` 只减去随机基准并按样本量缩放。

该指标可以用于比较同一个数据集上的不同模型，但不建议单独用它比较不同数据集。

---

## 9. `uplift_curve`：Uplift Gain 曲线

在排名前 \(k\) 个用户中，代码计算：

\[
U(k)
=
k\left(
\frac{Y_t(k)}{N_t(k)}
-
\frac{Y_c(k)}{N_c(k)}
\right)
\]

括号内是：

```text
treatment 组结果率 - control 组结果率
```

例如：

```text
treatment 购买率 = 10%
control 购买率   = 6%
```

则该人群的观测 uplift 是 4 个百分点。

Qini Curve 和 Uplift Curve 都评估 CATE 排序，但缩放方式不同：

- Qini gain 以 treatment 人数为主要尺度；
- Uplift gain 用选中的总人数乘以两组结果率差。

---

## 10. `_perfect_uplift_curve`：理想 Uplift 曲线

该函数根据观测到的 `Y` 和 `T` 构造一个理想排序分数，用于归一化 Uplift AUC。

它不是某个 baseline 模型的预测，也不会参与训练，只是评估时的参考上界曲线。

---

## 11. `uplift_auc_normalized`：归一化 Uplift AUC

计算方式是：

\[
\text{Normalized Uplift AUC}
=
\frac{
AUC_{model}-AUC_{random}
}{
AUC_{perfect}-AUC_{random}
}
\]

它总结整条 Uplift Curve，而不是只检查前 10% 或前 20% 的用户。

基本解读为：

- 越大通常越好；
- 0 附近表示接近随机排序；
- 负值表示排序较差。

建议将它作为 `qini_auc_normalized` 的补充指标，检查两种曲线是否得到相似的模型排名。

---

## 12. `uplift_at_k`：目标人群的观测 uplift

`uplift_at_k` 先根据预测 CATE 从高到低排序，然后只选择排名最高的前一部分用户。

例如：

```python
uplift_at_k(y, score, treatment, 0.10)
```

它计算：

\[
\text{Uplift@10\%}
=
\bar Y_{treatment,top10\%}
-
\bar Y_{control,top10\%}
\]

假设前 10% 用户中：

```text
treatment 购买率 = 3.2%
control 购买率   = 2.1%
```

那么：

```text
Uplift@10% = 0.032 - 0.021 = 0.011
```

这表示模型排名前 10% 的用户中，treatment 组的观测购买率比 control 组高 1.1 个百分点。

本项目输出：

- `uplift_at_10pct`：适合分析只干预前 10% 用户的情况；
- `uplift_at_20pct`：适合分析只干预前 20% 用户的情况。

### 边界情况

如果选中的前 10% 或前 20% 用户只包含 treatment，或只包含 control，就无法计算两组结果率之差，代码会返回 `NaN`。

这种情况更容易在小样本测试中出现。

如果恰好在 10% 或 20% 截止位置有多个相同 CATE，当前代码会根据原始数据行顺序选择其中一部分。因此 Uplift@K 建议用作商业意义较强的辅助指标，主要模型选择仍以整条曲线的 Qini AUC 为主。

---

## 13. `auuc`：Uplift Curve 下的原始面积

当前代码定义：

\[
AUUC
=
\frac{AUC_{uplift\ curve}}{N^2}
\]

它没有减去随机排序基准线。

因此需要注意：

> 随机排序或常数 CATE 预测的 AUUC 不一定等于 0。

如果整个实验本来就存在正的平均干预效果，即使随机排列用户，Uplift Curve 也可能具有正面积。

在同一个数据集中，所有模型使用完全相同的评估用户和整体干预效果，因此 AUUC 仍然可以用于辅助比较。

---

## 14. `evaluate_uplift`：统一评估入口

`evaluate_uplift` 返回一个字典，共包含九项输出：

| 指标 | 主要作用 | 是否越大越好 |
|---|---|---:|
| `qini_auc_normalized` | 主要 uplift 排序指标，当前调参 objective | 是 |
| `qini_coefficient` | Qini 曲线超过随机基准线的面积 | 是 |
| `uplift_auc_normalized` | 整体 Uplift Curve 排序能力 | 是 |
| `auuc` | 原始 Uplift Curve 面积 | 同一数据集内越大越好 |
| `uplift_at_10pct` | 排名前 10% 用户的观测 uplift | 是 |
| `uplift_at_20pct` | 排名前 20% 用户的观测 uplift | 是 |
| `ate_observed` | 整个评估集的 treatment/control 结果率差 | 不是模型排序指标 |
| `cate_mean` | 模型预测 CATE 的平均值 | 否 |
| `cate_std` | 模型预测 CATE 的标准差 | 否 |

---

## 15. `ate_observed`：观测到的总体平均差

计算方式是：

\[
\text{ATE}_{observed}
=
\bar Y_{T=1}
-
\bar Y_{T=0}
\]

它表示整个评估集中，treatment 组与 control 组的平均结果差。

在随机实验数据中，它可以作为总体平均干预效果的简单估计。

但是：

- 它由 `Y` 和 `T` 直接计算；
- 它不使用模型预测的 `score`；
- 在同一个评估集上，所有模型的 `ate_observed` 都相同。

因此不能用它选择模型或调节超参数。

对非随机的观察数据，两组用户可能本来就不同，这个简单差值不能直接解释成因果 ATE，需要 propensity weighting 等调整方法。

---

## 16. `cate_mean` 和 `cate_std`：预测分布诊断

### `cate_mean`

```python
np.mean(score)
```

它是模型预测 CATE 的平均值。

可以将它与 `ate_observed` 进行粗略比较：

- 两者接近：模型的平均预测可能比较合理；
- 差距很大：模型可能存在整体校准偏差。

但两者接近不代表排序一定正确。

### `cate_std`

```python
np.std(score)
```

它表示不同用户的预测 CATE 有多少差异。

- 非常接近 0：模型几乎给所有用户相同的分数；
- 较大：模型认为用户之间存在明显的效果异质性；
- 过大：也可能是模型过拟合或预测过于极端。

`cate_std` 不是越大越好，只能作为诊断信息。

---

## 17. 实际调参和模型选择建议

对 T-Learner、X-Learner、DR-Learner 和 DragonNet，建议使用以下优先级：

1. 使用 validation `qini_auc_normalized` 作为主要调参 objective；
2. 检查 `uplift_auc_normalized` 是否支持相似结论；
3. 使用 `uplift_at_10pct` 和 `uplift_at_20pct` 判断有限预算下的目标人群价值；
4. 用 `cate_mean`、`cate_std` 和 `ate_observed` 检查预测分布是否明显异常；
5. 超参数选择完成后，只在独立 test 上评估最终模型；
6. 不应根据 test 结果继续反复调参。

不能只因为某次 validation Qini 略高就认为必然更好。对低转化数据，少量正例变化就可能造成指标波动。

判断是否停止调参时，建议同时检查：

- 更多 trial 是否已经不再带来稳定提升；
- 主指标和辅助指标是否方向一致；
- 多个随机种子或 bootstrap 结果是否稳定；
- 提升是否大于统计波动。

---

## 18. 使用前提和局限

### 18.1 当前只支持二元结果

`metrics.py` 要求 `Y` 是 `0/1`，因此可以直接评估 `conversion` 和 `visit`，但不能直接评估连续的 `spend`。

### 18.2 当前只支持二元干预

`T` 必须是 `0/1`。如果以后同时比较多种邮件、多种优惠券或多个剂量，需要扩展为 multi-treatment 评估。

### 18.3 因果解释依赖随机化或无混杂假设

当 treatment 不是随机分配时，treatment/control 结果率差可能包含混杂偏差。

这时不能只依赖本文件中的简单两组差异，而需要 propensity score、IPW 或 doubly robust 评估等方法。

### 18.4 当前没有输出置信区间

当前代码只输出指标点估计，没有输出 bootstrap 标准误或置信区间。

因此，模型 A 的 Qini 比模型 B 高 0.01，不一定代表 A 真实地更好。置信区间、多个随机种子和稳定性检验属于项目后续的 robustness tests。

---

## 19. 代码检查结论

对当前版本的检查结果如下：

- Qini Curve 核心公式正确；
- Uplift Curve 核心公式正确；
- 随机基准面积处理正确；
- 归一化 Qini AUC 和 Uplift AUC 的计算流程与 scikit-uplift 定义一致；
- 相同 CATE 分数在曲线中被分组处理；
- 输入长度、NaN、二元标签和 treatment arm 检查完整；
- 项目完整测试结果为 `23 passed, 1 skipped`。

当前没有发现会导致四个 baseline 模型对比结论直接错误的核心公式问题，可以继续用于 T-Learner、X-Learner、DR-Learner 和 DragonNet 的调参与评估。

需要谨慎解释的主要是：

1. 小样本中的 Qini 波动；
2. Uplift@K 中缺少某个 treatment arm 时的 `NaN`；
3. AUUC 没有减去随机基准；
4. `ate_observed`、`cate_mean` 和 `cate_std` 不是模型排序性能指标；
5. 当前结果不包含置信区间。

