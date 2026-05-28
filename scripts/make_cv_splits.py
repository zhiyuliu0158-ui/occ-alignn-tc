"""Create grouped cross-validation CSVs for Tc training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _group_series(df: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column in df.columns:
        groups = df[group_column].fillna("").astype(str)
        fallback = df.get("sample_id", pd.Series(np.arange(len(df)), index=df.index)).astype(str)
        return groups.where(groups.str.len() > 0, fallback)
    if "sample_id" in df.columns:
        return df["sample_id"].astype(str)
    return pd.Series(np.arange(len(df)).astype(str), index=df.index)


def _make_fold_assignments(groups: pd.Series, n_folds: int, seed: int) -> dict[str, int]:
    unique_groups = np.asarray(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    return {group: int(index % n_folds) for index, group in enumerate(unique_groups)}


def make_cv_splits(
    df: pd.DataFrame,
    n_folds: int,
    group_column: str,
    val_group_frac: float,
    seed: int,
) -> list[pd.DataFrame]:
    """Return DataFrames with fold-specific train/val/test split labels."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    groups = _group_series(df, group_column)
    assignments = _make_fold_assignments(groups, n_folds=n_folds, seed=seed)
    rng = np.random.default_rng(seed + 1009)
    outputs: list[pd.DataFrame] = []
    for fold in range(n_folds):
        out = df.copy()
        test_groups = {group for group, group_fold in assignments.items() if group_fold == fold}
        remaining_groups = sorted(set(assignments) - test_groups)
        n_val = max(1, int(round(len(remaining_groups) * val_group_frac)))
        shuffled_remaining = np.asarray(remaining_groups)
        rng.shuffle(shuffled_remaining)
        val_groups = set(shuffled_remaining[:n_val])
        split = []
        for group in groups:
            if group in test_groups:
                split.append("test")
            elif group in val_groups:
                split.append("val")
            else:
                split.append("train")
        out["split"] = split
        out["cv_fold"] = fold
        outputs.append(out)
    return outputs


def _summary_row(df: pd.DataFrame, fold: int, group_column: str) -> dict[str, object]:
    row: dict[str, object] = {"fold": fold, "n_rows": len(df)}
    groups = _group_series(df, group_column)
    row["n_groups"] = groups.nunique()
    for split in ("train", "val", "test"):
        mask = df["split"].astype(str) == split
        row[f"{split}_rows"] = int(mask.sum())
        row[f"{split}_groups"] = int(groups[mask].nunique())
        if "Tc_K" in df.columns:
            tc = pd.to_numeric(df.loc[mask, "Tc_K"], errors="coerce")
            row[f"{split}_tc_gt_120"] = int((tc > 120.0).sum())
        if "formula_standardized" in df.columns:
            row[f"{split}_h3s"] = int((df.loc[mask, "formula_standardized"].astype(str) == "H3S").sum())
    return row


def _rewrite_relative_paths(df: pd.DataFrame, source_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Keep relative file paths valid after writing split CSVs to output_dir."""
    if "cif_path" not in df.columns:
        return df
    out = df.copy()

    def rewrite(value: object) -> object:
        if pd.isna(value):
            return value
        path_text = str(value)
        path = Path(path_text)
        if path.is_absolute():
            return path_text
        source_path = source_dir / path
        if not source_path.exists():
            return path_text
        return os.path.relpath(source_path, output_dir)

    out["cif_path"] = out["cif_path"].map(rewrite)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--group_column", default="parent_cif_id")
    parser.add_argument("--val_group_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    df = pd.read_csv(data_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_frames = make_cv_splits(
        df,
        n_folds=args.n_folds,
        group_column=args.group_column,
        val_group_frac=args.val_group_frac,
        seed=args.seed,
    )
    summary = []
    for fold, fold_df in enumerate(fold_frames):
        fold_df = _rewrite_relative_paths(fold_df, data_csv.parent, out_dir)
        fold_df.to_csv(out_dir / f"fold_{fold}.csv", index=False)
        summary.append(_summary_row(fold_df, fold=fold, group_column=args.group_column))
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
