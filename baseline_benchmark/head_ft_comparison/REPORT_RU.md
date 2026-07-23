# CausalPFN: zero-shot против head-only fine-tuning

## Протокол

- Датасеты: Hillstrom, LZD, RetailHero.
- Полный объём данных (`max_rows=0`).
- Разбиение: `validation`, seed 42.
- Test outcomes не использовались.
- Zero-shot результаты взяты из ранее рассчитанного запуска с тем же
  разбиением.
- Для `causalpfn_head_ft`: 10 эпох, 8 задач на эпоху, context 1024,
  query 256, learning rate `1e-4`, 5-fold DR pseudo-labels.
- Для inference обе модели используют context/query limits 4096 и 1024
  соседей.
- Совпадение evaluation `epk_id`, treatment и outcome проверено для каждой
  пары моделей.

## Результаты

| Dataset | Zero-shot Qini | Head-FT Qini | Delta Qini | Zero-shot AUUC | Head-FT AUUC | Delta Uplift@10% | Delta Uplift@20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hillstrom | -0.105466 | -0.133124 | -0.027659 | 0.002233 | 0.002157 | -0.001513 | -0.002164 |
| LZD | 0.106970 | 0.093399 | -0.013571 | 0.004607 | 0.004369 | +0.002843 | -0.005458 |
| RetailHero | 0.009546 | 0.009662 | +0.000116 | 0.018607 | 0.018650 | +0.004235 | -0.005144 |

## Диагностика ранжирования

| Dataset | Spearman CATE | Совпадение top-10% | Норма изменения head |
|---|---:|---:|---:|
| Hillstrom | 0.927955 | 48.83% | 1.8919 |
| LZD | 0.925774 | 89.33% | 2.5616 |
| RetailHero | 0.986624 | 91.42% | 3.1237 |

Inner-validation loss снижался на всех трёх датасетах, поэтому оптимизация
head работает технически корректно. Однако улучшение surrogate loss на
DR potential-outcome labels не переносится устойчиво на основной uplift
ranking: Qini ухудшился на Hillstrom и LZD, а улучшение RetailHero мало и
может находиться в пределах single-seed вариативности.

Текущую конфигурацию не следует переносить на test. Следующий validation
эксперимент целесообразно ограничить более консервативными режимами:
learning rate `1e-5`, 3--5 эпох и L2-SP regularization относительно исходного
pretrained head. После выбора одной конфигурации результат нужно проверить
на нескольких validation seeds, и только затем один раз запустить test.
