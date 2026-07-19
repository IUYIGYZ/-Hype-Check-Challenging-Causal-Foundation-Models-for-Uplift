# Baseline Benchmark Data Processing Guide

## 1. Purpose of data.py

The data-processing code is located in **baseline_benchmark/data.py**.

Its only responsibility is to convert cleaned datasets into consistent numerical training data that all baseline models can use fairly. It does not train a model or calculate final evaluation metrics.

The overall workflow is:

~~~text
Cleaned Parquet data
        ↓
Load and align X / T / Y
        ↓
Remove IDs, treatment time, and other non-feature columns
        ↓
Create train / validation / test splits with a fixed random seed
        ↓
Learn imputation, scaling, and one-hot rules from the training set only
        ↓
Apply the same rules to validation and test data
        ↓
Return a PreparedData object
        ↓
Give every baseline model exactly the same processed data
~~~

## 2. Meaning of X, T, and Y

| Symbol | Name | Meaning | Example |
|---|---|---|---|
| X | User covariates | Information known before treatment | Purchase history, channel, and geographic category |
| T | Treatment | Whether the user received the intervention | T=1 treated and T=0 untreated |
| Y | Outcome | The result that treatment is intended to change | Conversion or purchase |

The project does not merely predict Y. It estimates the difference between the outcome under treatment and the outcome without treatment:

~~~text
uplift(X) = P(Y=1 | X,T=1) - P(Y=1 | X,T=0)
~~~

A larger uplift means that the user is more likely to convert because of the intervention.

## 3. Supported datasets

| Code name | Cleaned directory | Default Y | Splitting method |
|---|---|---|---|
| criteo | Criteo-ITE-v2.1 | conversion | Duplicate-feature-vector group split |
| hillstrom | Hillstrom | conversion | T×Y stratified split |
| lzd | LZD | Y | Duplicate-feature-vector group split |
| retailhero | Retailhero-uplift | Y | T×Y stratified split |

Orange Telecom Churn is not currently included in the formal baseline configuration because the source of its Treatment variable and its dataset version still require further confirmation.

### 3.1 Outcome selection

Different Outcomes answer different questions. They should be evaluated in separate experiments and must not be mixed within one experiment.

| Outcome | Type | Advantage | Limitation | Current use |
|---|---|---|---|---|
| conversion | Binary | Closest to the final purchase objective | Few positive events, causing larger Qini/AUUC variation | Primary Outcome for Criteo and Hillstrom |
| visit | Binary | More events and usually more stable with small samples | A visit does not necessarily mean a purchase | Optional Secondary Outcome for Criteo/Hillstrom |
| spend | Continuous | Represents monetary and economic value | Not a binary Y | Available in Hillstrom but unsupported by the current benchmark |
| Y | Binary | Main result already defined by the dataset | Business meaning depends on the dataset documentation | Primary Outcome for LZD and RetailHero |

The Primary Outcome is used for the main result table and conclusions. A Secondary Outcome provides supplementary analysis. The current main experiments use conversion for Hillstrom and Criteo:

~~~text
uplift(X) = P(conversion=1 | X,T=1) - P(conversion=1 | X,T=0)
~~~

The **lag** column in outcomes.parquet is a technical or time-related auxiliary field, not the current Outcome. Criteo's **exposure** indicates whether treatment was actually observed and may be a post-treatment variable. It should not directly replace conversion as the Primary Outcome.

## 4. Cleaned data structure

Each dataset requires at least two files:

~~~text
<dataset>/
├── features.parquet
└── outcomes.parquet
~~~

**features.parquet** contains:

- epk_id: the sample identifier;
- T: the Treatment label;
- other pre-treatment covariates X.

**outcomes.parquet** contains:

- epk_id: used to align the outcome table with the feature table;
- one or more Outcome columns, such as conversion.

Therefore, T is stored in features.parquet, not outcomes.parquet.

The code checks whether the two files have the same number of rows and whether their epk_id values are aligned row by row. Processing stops if the ID order differs. This prevents the covariates of one user from being incorrectly combined with another user's Outcome.

## 5. Loading and optional sampling

### 5.1 Batch loading

Criteo contains nearly 14 million rows. **_read_selected_rows()** uses PyArrow to read Parquet data in batches of up to 131,072 rows, avoiding unnecessary full-table materialization in memory.

### 5.2 max_rows

By default, **prepare_data()** uses at most 50,000 rows for quick development and debugging. If max_rows is smaller than the dataset, the code samples without replacement using the specified fixed seed.

The same dataset, max_rows, and seed produce the same sample, making model comparisons reproducible.

max_rows can be used for an overall data-scale stress test, but it is not the same as the CausalPFN context size:

- max_rows controls the total number of observations used in the benchmark;
- context size controls how many context or support samples CausalPFN sees for one prediction.

Context size belongs to the CausalPFN model-input stage and is not directly implemented by the current data.py.

A 50,000-row sample is suitable for debugging but not necessarily for final experiments. Positive conversion events are very rare in Criteo, so formal experiments should use a substantially larger sample or the full dataset.

## 6. Columns excluded from X

The code always excludes:

~~~text
epk_id
T
treatment_dt
split
~~~

Reasons:

- epk_id is an identifier, not a meaningful predictive covariate;
- T is retained as a separate Treatment input and must not be duplicated inside X;
- treatment_dt is the time at which the user received treatment. It may contain treatment or post-treatment information and creates a leakage risk;
- split only records whether a row belongs to the training, validation, or test set.

## 7. Validation of T and Y

The current benchmark supports only binary Treatment and binary Outcome variables.

The code requires:

~~~text
T must contain both 0 and 1
Y must contain both 0 and 1
~~~

If only the treated group or only the control group is present, the treatment effect cannot be estimated. If Y is entirely zero or entirely one, there is no Outcome variation to learn.

## 8. Train, validation, and test splitting

The default proportions are:

| Split | Default proportion | Purpose |
|---|---:|---|
| Train | 60% | Fit the model and learn preprocessing rules |
| Validation | 20% | Model selection, calibration, and hyperparameter tuning |
| Test | 20% | Final fair evaluation |

### 8.1 Hillstrom and RetailHero: stratified splitting

These datasets use StratifiedShuffleSplit, preferably stratifying by the four T×Y combinations:

~~~text
T=0, Y=0
T=0, Y=1
T=1, Y=0
T=1, Y=1
~~~

The objective is to keep the Treatment-arm and positive-Outcome proportions similar across training, validation, and test data. If a T×Y combination has too few observations, the code falls back to stratification by T only.

Many Hillstrom users have identical feature combinations because its covariates are coarse, such as geographic category, channel, and purchase-history interval. This does not necessarily mean that the same user appears repeatedly. Therefore, Hillstrom does not use feature-vector grouping.

### 8.2 Criteo and LZD: group-safe splitting

For these datasets, the code hashes the complete X vector and uses that hash as a group in GroupShuffleSplit.

If two rows have exactly the same X, both must enter the same split. They cannot be divided between training and test data.

This group is a complete-feature-vector group, not a user ID. It reduces the risk that duplicated feature vectors leak information across splits.

Because a group cannot be divided, final row counts may not be exactly 60%/20%/20%. Small differences from the requested proportions are expected.

## 9. Post-split validation

**_validate_splits()** checks that:

1. training, validation, and test data do not overlap;
2. the three splits cover every selected observation;
3. every split contains both T=0 and T=1;
4. each Treatment arm in every split has enough positive Y=1 observations.

If one Treatment arm has fewer than 10 positive Outcomes, the code emits:

~~~text
Qini/AUUC estimates will be unstable
~~~

This warning does not necessarily mean that the code is incorrect. It means the current sample is too small for stable Qini/AUUC estimates. Formal experiments should use more data and multiple random seeds.

## 10. Numerical feature processing

Numerical features go through two steps.

### 10.1 Median imputation

Missing numerical values are replaced with the median of that column in the training set.

Example:

~~~text
Original age: 20, 25, NaN, 80
Training-set median: 25
After imputation: 20, 25, 25, 80
~~~

The median is relatively insensitive to extreme values.

### 10.2 Standardization

After imputation, StandardScaler applies:

~~~text
x_scaled = (x - training-set mean) / training-set standard deviation
~~~

This places numerical features on comparable scales and is especially useful for neural models.

## 11. Text and categorical feature processing

The code treats pandas columns with object, category, or string dtype as categorical features.

Typical categorical features in Hillstrom include:

~~~text
history_segment
zip_code
channel
~~~

They are processed in the following order.

### 11.1 Normalize missing markers

None, pd.NA, and NaN are converted into a common missing marker that SimpleImputer can recognize.

### 11.2 Most-frequent-category imputation

Suppose the training values of channel are:

~~~text
Web, Phone, Web, <missing>
~~~

The most frequent category is Web, so the missing value is replaced by Web.

### 11.3 One-Hot Encoding

Models cannot directly calculate with strings such as Web and Phone. Each category is converted into a separate binary column.

| Original channel | channel_Phone | channel_Web | channel_Multichannel |
|---|---:|---:|---:|
| Phone | 1 | 0 | 0 |
| Web | 0 | 1 | 0 |
| Multichannel | 0 | 0 | 1 |

These values do not represent category magnitude or order. They only indicate whether an observation belongs to a category.

### 11.4 Unseen validation or test categories

OneHotEncoder uses handle_unknown="ignore". If validation or test data contain a category never observed during training, transformation does not fail. The one-hot columns for that categorical feature are all zero for the unseen category.

## 12. Why preprocessing is fitted on training data only

The code executes:

~~~python
preprocessor.fit_transform(X_train)
preprocessor.transform(X_val)
preprocessor.transform(X_test)
~~~

Only the training set uses fit_transform. Validation and test data can only use rules learned from training data, including:

- numerical medians;
- numerical means and standard deviations;
- most frequent categorical values;
- the list of one-hot categories.

Learning these rules from the complete dataset would leak test-set information into the training process and make the final evaluation overly optimistic.

## 13. PreparedData output

**prepare_data()** returns a PreparedData object:

| Field | Meaning |
|---|---|
| X_train/X_val/X_test | float32 matrices after imputation, scaling, and one-hot encoding |
| t_train/t_val/t_test | Treatment labels |
| y_train/y_val/y_test | Outcome labels |
| id_train/id_val/id_test | IDs retained for tracking but not passed as model features |
| feature_names | Names of transformed numerical and one-hot columns |
| split_table | The split assigned to every epk_id |
| preprocessor | The fitted transformer learned from training data |
| group_safe | Whether the dataset uses group-safe splitting |

After transformation, the code verifies that feature matrices contain no NaN or infinite values. Processing stops if a non-finite value remains.

## 14. Fair baseline comparison

Within one baseline experiment, prepare_data() runs once.

Every model receives the same:

- dataset;
- Outcome definition;
- sampled observations;
- train/validation/test split;
- transformed feature matrices;
- T and Y labels;
- evaluation metrics.

Only the learning method differs. Results from S-Learner, T-Learner, X-Learner, DR-Learner, TARNet, and DragonNet can therefore be compared fairly.

## 15. Direct usage

~~~python
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
~~~

To use the baseline entry point:

~~~bash
cd 项目/baseline_benchmark

python run_baselines.py \
  --dataset hillstrom \
  --models constant_ate,s_learner,t_learner,x_learner,dr_learner \
  --max-rows 50000 \
  --seed 42
~~~

## 16. Current status

The data pipeline has passed the automated tests. It has also been dry-run on 5,000 rows from each of Hillstrom, RetailHero, LZD, and Criteo. All four datasets produced training, validation, and test matrices with consistent feature dimensions and no NaN or infinite values.

The current data-processing workflow is ready for the first baseline model-comparison runs. Formal result generation should additionally:

1. use larger samples for datasets with rare Outcomes;
2. run multiple random seeds;
3. report the mean, standard deviation, or confidence interval of each metric;
4. save and reuse the same data splits and fitted preprocessors.
