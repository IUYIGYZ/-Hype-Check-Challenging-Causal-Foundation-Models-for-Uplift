# Модели поправки CausalPFN

Реализованы две экспериментальные модели:

- `causalpfn_ridge_correction`;
- `causalpfn_hgb_correction`.

Обе оставляют официальный pretrained CausalPFN неизменным и обучают отдельную
модель residual-поправки.

## Формула

```text
CATE_final = CATE_CausalPFN + strength * correction
```

Цель correction-модели:

```text
residual = cross-fitted DR effect - OOF CausalPFN CATE
```

Correction-модель получает признаки:

```text
[X, OOF CATE_CausalPFN, OOF mu0, OOF mu1]
```

Для каждой строки OOF CausalPFN использует context, который не содержит fold
этой строки. Nuisance outcome-модели строятся на тех же train-folds.

## Ridge

`causalpfn_ridge_correction` использует:

```text
StandardScaler
Ridge(alpha=10)
```

Это основной безопасный вариант с сильной регуляризацией и небольшим риском
переобучения.

## HGB

`causalpfn_hgb_correction` использует ограниченный
`HistGradientBoostingRegressor`:

```text
max_iter=50
learning_rate=0.03
max_leaf_nodes=15
min_samples_leaf=200
l2_regularization=1.0
early_stopping=False
```

HGB может нелинейно изменить ranking, но чувствительнее к шуму DR-target.

## Основные параметры CLI

| Параметр | Default | Назначение |
|---|---:|---|
| `--causalpfn-correction-strength` | 0.5 | Сила добавляемой поправки |
| `--causalpfn-correction-folds` | 3 | Число OOF folds |
| `--causalpfn-correction-winsor-quantile` | 0.01 | Обрезка крайних residual-target |
| `--causalpfn-correction-center` | false | Центрировать среднюю train-поправку |
| `--causalpfn-correction-ridge-alpha` | 10 | Регуляризация Ridge |
| `--causalpfn-correction-max-iter` | 50 | Число итераций HGB |
| `--causalpfn-correction-learning-rate` | 0.03 | Learning rate HGB |
| `--causalpfn-correction-max-leaf-nodes` | 15 | Сложность деревьев HGB |
| `--causalpfn-correction-min-samples-leaf` | 200 | Минимальный лист HGB |
| `--causalpfn-correction-l2-regularization` | 1.0 | L2 HGB |

`strength=0` является контрольным режимом: итоговый CATE точно совпадает с
zero-shot CausalPFN.

## Пример

```powershell
.\.venv\Scripts\python.exe baseline_benchmark\run_baselines.py `
  --cleaned-root "C:\path\to\data_A_cleaned" `
  --dataset lzd `
  --models causalpfn,causalpfn_ridge_correction,causalpfn_hgb_correction `
  --evaluation-split validation `
  --seed 42 `
  --causalpfn-correction-folds 3 `
  --causalpfn-correction-strength 0.5
```

## Артефакты

Помимо обычных `metrics.csv` и `predictions.parquet`, сохраняются:

```text
causalpfn_ridge_correction_training.json
causalpfn_hgb_correction_training.json
causalpfn_ridge_correction_components.parquet
causalpfn_hgb_correction_components.parquet
```

Они содержат:

- диапазон raw и winsorized residual;
- propensity и фактическое число folds;
- стандартное отклонение OOF CausalPFN;
- корреляцию OOF CausalPFN с DR-effect;
- средний размер и дисперсию поправки;
- корреляцию исходного и скорректированного CATE.

Файлы `components.parquet` сохраняют для каждого evaluation-объекта исходный
`base_cate`, предсказанную `correction`, её силу и итоговый `cate_pred`.

OOF CausalPFN вычислительно дорог: для трёх folds каждая correction-модель
строит три fold-контекста и один финальный полный context. Сначала следует
подбирать параметры на validation и ограниченном числе seeds. Test используется
только после фиксации конфигурации.
