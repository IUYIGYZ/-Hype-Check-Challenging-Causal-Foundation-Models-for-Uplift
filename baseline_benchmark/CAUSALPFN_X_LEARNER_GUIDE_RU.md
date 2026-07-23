# X-Learner с CausalPFN

Модель называется `causalpfn_x_learner`. Она использует официальный
предобученный CausalPFN на первом этапе X-Learner и не изменяет его веса.

## Алгоритм

Обучающая выборка делится на `K` folds со стратификацией по сочетанию `T × Y`.
Для каждого fold CausalPFN получает context только из остальных folds и
предсказывает для отложенных строк:

```text
mu0(X) = E[Y(0) | X]
mu1(X) = E[Y(1) | X]
```

После объединения OOF-предсказаний строятся стандартные targets X-Learner:

```text
D0 = mu1(X) - Y  для контрольной группы
D1 = Y - mu0(X)  для treatment-группы
```

Второй CausalPFN получает объединённый непрерывный pseudo-outcome:

```text
D = D0, если T=0
D = D1, если T=1
```

Его потенциальные исходы интерпретируются как две effect-функции
`tau0(X)` и `tau1(X)`. Итоговая оценка:

```text
CATE(X) = p(T=1) * tau0(X) + (1 - p(T=1)) * tau1(X)
```

Официальный CausalPFN поддерживает непрерывные outcomes: перед передачей в
нейросеть они z-нормализуются отдельно по двум treatment-группам. Поэтому
первый и второй этапы этого варианта X-Learner используют CausalPFN.
Cross-fitting исключает утечку на этапе импутации: строка не входит в PFN
context, из которого получены её собственные `mu0` и `mu1`.

## Запуск

Из корня проекта:

```powershell
.\.venv\Scripts\python.exe baseline_benchmark\run_baselines.py `
  --cleaned-root "C:\Users\User\Downloads\Данные uplift\Данные uplift\data_A_cleaned" `
  --dataset hillstrom `
  --models causalpfn,x_learner,causalpfn_x_learner `
  --causalpfn-x-folds 3 `
  --evaluation-split validation
```

Доступные параметры:

- `--causalpfn-x-folds` — число PFN cross-fitting folds, по умолчанию 3;
- общие параметры `--causalpfn-*` управляют checkpoint, context/query,
  neighbours, device и cache.

Основные результаты записываются в `metrics.csv`, предсказания — в
`predictions.parquet`, а параметры OOF-эффектов — в
`causalpfn_x_learner_training.json`.

Обучение требует `K` отдельных CausalPFN для OOF-импутации и ещё один
CausalPFN для effect-функций, поэтому эксперимент примерно в `K + 1` раз
дороже одного zero-shot запуска.
