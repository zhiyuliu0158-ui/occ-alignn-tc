"""Tests for grouped cross-validation split generation."""

from __future__ import annotations

import pandas as pd

from scripts.make_cv_splits import make_cv_splits


def test_make_cv_splits_keeps_groups_exclusive() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "parent_cif_id": [f"g{i // 2}" for i in range(20)],
            "Tc_K": [float(i) for i in range(20)],
            "formula_standardized": ["H3S" if i < 2 else "MgB2" for i in range(20)],
        }
    )

    folds = make_cv_splits(df, n_folds=5, group_column="parent_cif_id", val_group_frac=0.2, seed=7)

    assert len(folds) == 5
    for fold_df in folds:
        groups_by_split = {
            split: set(fold_df.loc[fold_df["split"] == split, "parent_cif_id"])
            for split in ("train", "val", "test")
        }
        assert groups_by_split["train"].isdisjoint(groups_by_split["val"])
        assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
        assert groups_by_split["val"].isdisjoint(groups_by_split["test"])
        assert set(fold_df["split"]) == {"train", "val", "test"}
