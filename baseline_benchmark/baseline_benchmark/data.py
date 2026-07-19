from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


class DatasetSpec(TypedDict):
    directory: str
    outcome: str
    group_safe: bool

# 数据集说明
DATASET_SPECS: dict[str, DatasetSpec] = {
    # 对于outcome，例如 Criteo、Hillstrom 都可能有多个结果：
    # visit：是否访问网站； conversion：是否购买/转； spend：购买金额
    # 代码需要先确定：我们到底把什么当作“干预是否有效”的标准？
    "criteo": {
        "directory": "Criteo-ITE-v2.1",
        "outcome": "conversion",   # 默认设置，可以改
        # conversion的目标：预测“给用户发邮件后，是否更可能购买”，
        # i.e.uplift(X)=P(conversion=1|X,T=1)-P(conversion=1|X,T=0))
        "group_safe": True,
    },
    "hillstrom": {
        "directory": "Hillstrom",
        "outcome": "conversion",
        "group_safe": False,  #不需要特殊的重复样本分组划分
    },
    "lzd": {
        "directory": "LZD",
        "outcome": "Y",
        "group_safe": True,
    },
    "retailhero": {
        "directory": "Retailhero-uplift",
        "outcome": "Y",
        "group_safe": False,
    },
}

# 下面的列不能作为模型特征
NON_FEATURE_COLUMNS = {"epk_id", "T", "treatment_dt", "split"}


@dataclass
class PreparedData:
    dataset: str
    outcome: str
    feature_names: list[str]  
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    t_train: np.ndarray   # 干预标签
    t_val: np.ndarray
    t_test: np.ndarray
    y_train: np.ndarray   # 结果标签
    y_val: np.ndarray
    y_test: np.ndarray
    id_train: np.ndarray
    id_val: np.ndarray
    id_test: np.ndarray
    split_table: pd.DataFrame
    preprocessor: ColumnTransformer  # 数据处理规则
    group_safe: bool


def _read_selected_rows(   # 分批次读取数据（Criteo 数据很多，如果一次全部读进内存，可能很慢甚至爆内存。）
    path: Path,
    columns: list[str],
    selected: Optional[np.ndarray],
    batch_size: int = 131_072,  # 每次只读取 131072 = 2^17 个数据
) -> pd.DataFrame:
    """Read selected row positions without materializing a full large table."""
    parquet = pq.ParquetFile(path)
    if selected is None:
        return parquet.read(columns=columns).to_pandas()

    selected = np.asarray(selected, dtype=np.int64)
    pieces: list[pa.Table] = []
    offset = 0
    left = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        end = offset + len(batch)
        right = int(np.searchsorted(selected, end, side="left"))
        if right > left:
            local = selected[left:right] - offset
            pieces.append(pa.Table.from_batches([batch.take(pa.array(local))]))
        left = right
        offset = end
        if left == len(selected):
            break
    if not pieces:
        raise ValueError(f"No rows selected from {path}")
    return pa.concat_tables(pieces).to_pandas()


def load_cleaned_dataset(  # 读取并检查 X、T、Y
    cleaned_root: Path,
    dataset: str,
    outcome: Optional[str],
    max_rows: Optional[int],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str, bool]:
    """Load aligned X/T/Y from one cleaned dataset."""
    key = dataset.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASET_SPECS)}")
    spec = DATASET_SPECS[key]
    dataset_dir = cleaned_root / spec["directory"]
    feature_path = dataset_dir / "features.parquet"  # X 和 T
    outcome_path = dataset_dir / "outcomes.parquet"  # Y
    if not feature_path.exists() or not outcome_path.exists():
        raise FileNotFoundError(f"Missing cleaned Parquet files under {dataset_dir}")

    fpf = pq.ParquetFile(feature_path)
    opf = pq.ParquetFile(outcome_path)
    if fpf.metadata.num_rows != opf.metadata.num_rows:  # 检查 features 和 outcomes 的行数是否一致，若不一致，则无法对齐 X、T、Y
        raise ValueError("Feature and outcome row counts differ")
    n_rows = fpf.metadata.num_rows
    selected = None
    if max_rows is not None and max_rows > 0 and max_rows < n_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(n_rows, size=max_rows, replace=False))

    outcome_name = outcome or str(spec["outcome"])
    if outcome_name not in opf.schema_arrow.names:
        raise ValueError(
            f"Outcome {outcome_name!r} is absent from {outcome_path}; "
            f"available={opf.schema_arrow.names}"
        )
    feature_columns = [
        name for name in fpf.schema_arrow.names if name not in NON_FEATURE_COLUMNS
    ]
    feature_read_columns = ["epk_id", "T", *feature_columns]
    outcome_read_columns = ["epk_id", outcome_name]
    features = _read_selected_rows(feature_path, feature_read_columns, selected)
    outcomes = _read_selected_rows(outcome_path, outcome_read_columns, selected)

    if len(features) != len(outcomes):
        raise ValueError("Selected feature and outcome row counts differ")
    if not np.array_equal(features["epk_id"].to_numpy(), outcomes["epk_id"].to_numpy()):  # 检查id是否一致，防止乱序导致 X、T、Y 对不上
        raise ValueError("Feature and outcome IDs are not aligned row by row")

    X = features[feature_columns].copy()  # 分开 X、T、Y
    t = features["T"].to_numpy(dtype=np.int8)
    y = outcomes[outcome_name].to_numpy()
    ids = features["epk_id"].to_numpy()
    if not set(np.unique(t)).issubset({0, 1}) or len(np.unique(t)) != 2:  # 检查干预标签 T 是否为二分类且包含 0 和 1，T=0 表示未干预，T=1 表示干预
        raise ValueError("Treatment must contain both 0 and 1")
    if set(np.unique(y)) != {0, 1}:  # Y 必须是二分类，并且 0/1 都必须存在
        raise ValueError("Outcome must contain both 0 and 1")
    return X, t, y.astype(np.int8), ids, outcome_name, bool(spec["group_safe"])


def _strata(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    strata = np.char.add(t.astype(str), np.char.add("_", y.astype(str)))
    _, counts = np.unique(strata, return_counts=True)
    return strata if counts.min() >= 2 else t.astype(str)


def _split_indices(  # 数据集划分为训练集、验证集和测试集，目前支持两种划分方式：group_safe 和 stratified
    ## 对 Hillstrom 和 RetailHero，使用StratifiedShuffleSplit。尽量让三个部分都保持类似的：处理组比例，购买/未购买比例，这样会使得训练集、验证集和测试集的分布更接近真实情况，更可靠。
    ## 对 Criteo 和 LZD，使用GroupShuffleSplit。因为这些数据集可能存在重复的用户或其他分组特征，如果不考虑分组，可能会导致训练集和测试集之间存在泄漏，从而影响模型的泛化能力。
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    group_safe: bool,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to < 1")
    indices = np.arange(len(X))
    temp_fraction = val_fraction + test_fraction

    if group_safe:
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
        first = GroupShuffleSplit(n_splits=1, test_size=temp_fraction, random_state=seed)
        train_idx, temp_idx = next(first.split(indices, groups=groups))
        relative_test = test_fraction / temp_fraction
        second = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
        val_rel, test_rel = next(
            second.split(temp_idx, groups=groups[temp_idx])
        )
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
        for a, b, label in (
            (train_idx, val_idx, "train/val"),
            (train_idx, test_idx, "train/test"),
            (val_idx, test_idx, "val/test"),
        ):
            if np.intersect1d(groups[a], groups[b]).size:
                raise RuntimeError(f"Feature-vector group leakage detected across {label}")
        return train_idx, val_idx, test_idx

    first = StratifiedShuffleSplit(n_splits=1, test_size=temp_fraction, random_state=seed)
    train_idx, temp_idx = next(first.split(indices, _strata(t, y)))
    relative_test = test_fraction / temp_fraction
    second = StratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
    val_rel, test_rel = next(
        second.split(temp_idx, _strata(t[temp_idx], y[temp_idx]))
    )
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]



def _validate_splits(  # 检查数据划分是否合理：
                       # 1. train/val/test 是否互不相交且覆盖所有样本；
                       # 2. 每个划分中是否包含两种干预标签 T=0，T=1，即检查每一份数据是否还能支持 uplift/CATE 评估；
                       # 3. 每个划分中每个干预标签下的正样本数量是否足够，避免模型评估不稳定。（uplift = P(Y=1 | T=1) - P(Y=1 | T=0) 如果某一边正样本太少，转化率估计就会很不稳定。）
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> None:
    named = {"train": train_idx, "validation": val_idx, "test": test_idx}
    combined = np.concatenate([train_idx, val_idx, test_idx])
    if len(np.unique(combined)) != len(combined) or len(combined) != len(t):
        raise RuntimeError("Train/validation/test indices do not form a disjoint full partition")
    for split_name, idx in named.items():
        if set(np.unique(t[idx])) != {0, 1}:
            raise ValueError(f"{split_name} split does not contain both treatment arms")
        for arm in (0, 1):
            events = int(y[idx][t[idx] == arm].sum())
            if events < 10:
                warnings.warn(
                    f"{split_name} split has only {events} positive outcomes in treatment arm {arm}; "
                    "Qini/AUUC estimates will be unstable.",
                    RuntimeWarning,
                )


def _normalize_categorical_missing(values: object) -> object:
    """Convert every pandas-style missing marker to np.nan for SimpleImputer."""
    if isinstance(values, pd.DataFrame):
        normalized = values.astype(object).copy()
        return normalized.where(pd.notna(normalized), np.nan)
    normalized_array = np.asarray(values, dtype=object).copy()
    normalized_array[pd.isna(normalized_array)] = np.nan
    return normalized_array


def _make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:  # 把文字特征变成数字，用 One-Hot Encoding
    categorical = list(X_train.select_dtypes(include=["object", "category", "string"]).columns)
    numerical = [c for c in X_train.columns if c not in categorical]
    transformers = []
    if numerical:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),  # 对于数值列，缺失数字用中位数补。然后标准化，让数值变成均值附近、尺度更统一。
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "normalize_missing",
                            FunctionTransformer(
                                _normalize_categorical_missing,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="most_frequent")),   # 如果文本列有缺失值，用出现最多的类别补上
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),   # 用 One-Hot Encoding 把类别拆成 0/1 数字列；如果验证集/测试集出现训练集没见过的类别，直接忽略，防止报错。
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def prepare_data(
    cleaned_root: Path,
    dataset: str,
    outcome: Optional[str] = None,
    max_rows: Optional[int] = 50_000,
    seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> PreparedData:
    X, t, y, ids, outcome_name, group_safe = load_cleaned_dataset(
        cleaned_root=cleaned_root,
        dataset=dataset,
        outcome=outcome,
        max_rows=max_rows,
        seed=seed,
    )
    train_idx, val_idx, test_idx = _split_indices(
        X=X,
        t=t,
        y=y,
        group_safe=group_safe,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    _validate_splits(train_idx, val_idx, test_idx, t, y)
    preprocessor = _make_preprocessor(X.iloc[train_idx])   # 数据的转换规则 _make_preprocessor 只在训练集上学习，验证集和测试集只能用训练集学到的规则来转换，防止数据泄漏。
    X_train = np.asarray(preprocessor.fit_transform(X.iloc[train_idx]), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(X.iloc[val_idx]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X.iloc[test_idx]), dtype=np.float32)
    for name, matrix in (("train", X_train), ("val", X_val), ("test", X_test)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values remain in transformed {name} features")

    feature_names = preprocessor.get_feature_names_out().tolist()
    split = np.full(len(X), "", dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "validation", "test"
    split_table = pd.DataFrame({"epk_id": ids, "split": split})
    return PreparedData(
        dataset=dataset.lower(),
        outcome=outcome_name,
        feature_names=feature_names,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        t_train=t[train_idx],
        t_val=t[val_idx],
        t_test=t[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        id_train=ids[train_idx],
        id_val=ids[val_idx],
        id_test=ids[test_idx],
        split_table=split_table,
        preprocessor=preprocessor,
        group_safe=group_safe,
    )
