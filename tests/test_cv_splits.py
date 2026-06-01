"""Tests for grouped cross-validation split generation."""

from __future__ import annotations

import pandas as pd

from scripts.make_cv_splits import make_cv_splits, make_holdout_split, make_stratified_group_split


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


def test_make_stratified_group_split_keeps_groups_exclusive() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(24)],
            "parent_cif_id": [f"g{i // 2}" for i in range(24)],
            "family": ["cuprate_or_cu_oxide" if i % 3 == 0 else "other" for i in range(24)],
            "Tc_K": [float(i * 5) for i in range(24)],
            "pressure_GPa": [60.0 if i % 5 == 0 else 0.0 for i in range(24)],
            "formula_standardized": ["YBa2Cu3O7" if i % 3 == 0 else "FeSe" for i in range(24)],
        }
    )

    split = make_stratified_group_split(df, "parent_cif_id", train_frac=0.7, val_frac=0.15, seed=3)

    groups_by_split = {
        name: set(split.loc[split["split"] == name, "parent_cif_id"])
        for name in ("train", "val", "test")
    }
    assert groups_by_split["train"].isdisjoint(groups_by_split["val"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["val"].isdisjoint(groups_by_split["test"])


def test_make_holdout_split_moves_matching_groups_to_test() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "parent_cif_id": ["g1", "g1", "g2", "g3", "g4", "g4"],
            "family": ["magnesium_boride", "magnesium_boride", "other", "other", "cuprate_or_cu_oxide", "other"],
            "Tc_K": [39.0, 40.0, 5.0, 10.0, 92.0, 12.0],
            "pressure_GPa": [0.0] * 6,
            "formula_standardized": ["MgB2", "MgB2", "FeSe", "NbSe2", "YBa2Cu3O7", "FeSe"],
        }
    )

    split = make_holdout_split(
        df,
        group_column="parent_cif_id",
        holdout_column="family",
        holdout_value="magnesium_boride",
        val_group_frac=0.25,
        seed=1,
    )

    assert set(split.loc[split["parent_cif_id"] == "g1", "split"]) == {"test"}
    assert "test" not in set(split.loc[split["parent_cif_id"] != "g1", "split"])
