# Общая таблица экспериментов CausalPFN и uplift-бейзлайнов

## Протокол сравнения

- Датасеты: Hillstrom, LZD, RetailHero.
- Полный объём очищенных данных (`max_rows=0`).
- Seed: 42.
- Оценка: внешний `validation` split.
- Test split не использовался.
- Все модели сравниваются на совпадающих наблюдениях, treatment и outcome.
- `causalpfn` — официальный pretrained checkpoint без дообучения.
- `causalpfn_head_ft` — head-only fine-tuning с cross-fitted DR
  potential-outcome pseudo-labels.

## Основная таблица CausalPFN

| Dataset | Модель | N train | N validation | Qini normalized | Qini coefficient | Uplift AUC normalized | AUUC | Uplift@10% | Uplift@20% | CATE mean | CATE std | Fit, sec | Predict, sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hillstrom | CausalPFN zero-shot | 38 400 | 12 800 | -0.105466 | -0.000196 | -0.001647 | 0.002233 | 0.004129 | 0.009052 | 0.003073 | 0.006904 | 22.78 | 253.28 |
| Hillstrom | CausalPFN head-FT | 38 400 | 12 800 | -0.133124 | -0.000247 | -0.002094 | 0.002157 | 0.002616 | 0.006888 | 0.004511 | 0.003746 | 195.12 | 284.53 |
| LZD | CausalPFN zero-shot | 108 970 | 36 552 | 0.106970 | 0.000890 | 0.007275 | 0.004607 | 0.016677 | 0.021052 | 0.001281 | 0.018083 | 215.65 | 543.00 |
| LZD | CausalPFN head-FT | 108 970 | 36 552 | 0.093399 | 0.000777 | 0.006295 | 0.004369 | 0.019520 | 0.015594 | 0.001020 | 0.024322 | 385.33 | 458.40 |
| RetailHero | CausalPFN zero-shot | 120 023 | 40 008 | 0.009546 | 0.000980 | 0.007215 | 0.018607 | 0.047207 | 0.050681 | 0.034955 | 0.044365 | 31.77 | 919.27 |
| RetailHero | CausalPFN head-FT | 120 023 | 40 008 | 0.009662 | 0.000992 | 0.007371 | 0.018650 | 0.051442 | 0.045537 | 0.040168 | 0.059717 | 211.65 | 722.36 |

## Изменение после head-only fine-tuning

Положительная дельта означает улучшение относительно zero-shot.

| Dataset | Delta Qini | Delta AUUC | Delta Uplift@10% | Delta Uplift@20% | Spearman CATE | Совпадение top-10% | Норма изменения head |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hillstrom | -0.027659 | -0.000076 | -0.001513 | -0.002164 | 0.927955 | 48.83% | 1.8919 |
| LZD | -0.013571 | -0.000238 | +0.002843 | -0.005458 | 0.925774 | 89.33% | 2.5616 |
| RetailHero | +0.000116 | +0.000043 | +0.004235 | -0.005144 | 0.986624 | 91.42% | 3.1237 |

## CausalPFN относительно лучшего классического бейзлайна

Лучший бейзлайн выбран отдельно на каждом датасете по основному показателю
`qini_auc_normalized`.

| Dataset | Лучший бейзлайн | Qini бейзлайна | CausalPFN zero-shot | CausalPFN head-FT | Отставание head-FT от лучшего |
|---|---|---:|---:|---:|---:|
| Hillstrom | DR-Learner | 0.304109 | -0.105466 | -0.133124 | -0.437233 |
| LZD | X-Learner | 0.110729 | 0.106970 | 0.093399 | -0.017331 |
| RetailHero | DragonNet | 0.018551 | 0.009546 | 0.009662 | -0.008889 |

## Все модели: Qini и targeting-метрики

| Dataset | Модель | Qini normalized | AUUC | Uplift@10% | Uplift@20% |
|---|---|---:|---:|---:|---:|
| Hillstrom | DR-Learner | 0.304109 | 0.003370 | 0.007212 | 0.008529 |
| Hillstrom | X-Learner | 0.292885 | 0.003322 | 0.011332 | 0.006794 |
| Hillstrom | DragonNet | 0.099239 | 0.002794 | -0.000636 | 0.003808 |
| Hillstrom | T-Learner | 0.080849 | 0.002732 | 0.009439 | 0.001945 |
| Hillstrom | CausalPFN zero-shot | -0.105466 | 0.002233 | 0.004129 | 0.009052 |
| Hillstrom | CausalPFN head-FT | -0.133124 | 0.002157 | 0.002616 | 0.006888 |
| LZD | X-Learner | 0.110729 | 0.004687 | 0.033040 | 0.022485 |
| LZD | DR-Learner | 0.109086 | 0.004639 | 0.019128 | 0.018016 |
| LZD | CausalPFN zero-shot | 0.106970 | 0.004607 | 0.016677 | 0.021052 |
| LZD | CausalPFN head-FT | 0.093399 | 0.004369 | 0.019520 | 0.015594 |
| LZD | T-Learner | 0.082082 | 0.004226 | 0.024669 | 0.016130 |
| LZD | DragonNet | 0.040659 | 0.003478 | 0.015405 | 0.011503 |
| RetailHero | DragonNet | 0.018551 | 0.020496 | 0.053362 | 0.051988 |
| RetailHero | X-Learner | 0.018145 | 0.020367 | 0.059319 | 0.053899 |
| RetailHero | DR-Learner | 0.016922 | 0.020132 | 0.050164 | 0.047776 |
| RetailHero | CausalPFN head-FT | 0.009662 | 0.018650 | 0.051442 | 0.045537 |
| RetailHero | CausalPFN zero-shot | 0.009546 | 0.018607 | 0.047207 | 0.050681 |
| RetailHero | T-Learner | 0.006139 | 0.017850 | 0.039062 | 0.044057 |

## Итог

Head-only fine-tuning технически обучается и снижает внутренний loss, но
текущая конфигурация не улучшает CausalPFN устойчиво:

- основной Qini ухудшается на Hillstrom и LZD;
- на RetailHero улучшение Qini составляет только `0.000116`;
- Uplift@10% улучшается на LZD и RetailHero, но Uplift@20% одновременно
  ухудшается;
- перенос текущей конфигурации на test не рекомендуется.

Следующий эксперимент должен использовать более консервативное обновление
head: learning rate `1e-5`, 3--5 эпох и L2-SP regularization относительно
исходных pretrained-весов, после чего конфигурацию следует проверить на
нескольких validation seeds.

## Источники внутри проекта

- `calculated_metrics/all_metrics.csv` — исходные результаты бейзлайнов и
  zero-shot CausalPFN.
- `head_ft_comparison/*/metrics.csv` — результаты head-only fine-tuning.
- `head_ft_comparison/comparison_seed42_validation.csv` — парные дельты и
  диагностика ранжирования.
