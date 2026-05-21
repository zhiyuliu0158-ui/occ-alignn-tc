"""Train/validation/test split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_split_values(series: pd.Series) -> bool:
    values = set(series.dropna().astype(str).str.lower())
    return {"train", "val", "test"}.issubset(values)


def ensure_split(
    dataframe: pd.DataFrame,
    split_column: str = "split",
    group_column: str = "parent_cif_id",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Use an existing split column or create grouped train/val/test splits."""
    df = dataframe.copy()
    if split_column in df.columns and _valid_split_values(df[split_column]):
        df[split_column] = df[split_column].astype(str).str.lower()
        return df

    if group_column in df.columns:
        groups = df[group_column].fillna("").astype(str)
        fallback = df.get("sample_id", pd.Series(np.arange(len(df)), index=df.index)).astype(str)
        groups = groups.where(groups.str.len() > 0, fallback)
    elif "sample_id" in df.columns:
        groups = df["sample_id"].astype(str)
    else:
        groups = pd.Series(np.arange(len(df)).astype(str), index=df.index)

    unique_groups = np.asarray(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_train = max(1, int(round(n_groups * train_frac)))
    n_val = max(1, int(round(n_groups * val_frac))) if n_groups >= 3 else 0
    if n_train + n_val >= n_groups and n_groups > 1:
        n_train = max(1, n_groups - 2)
        n_val = 1
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])
    if not test_groups:
        test_groups = val_groups.copy()
        val_groups = set()

    splits = []
    for group in groups:
        if group in train_groups:
            splits.append("train")
        elif group in val_groups:
            splits.append("val")
        else:
            splits.append("test")
    df[split_column] = splits
    return df
