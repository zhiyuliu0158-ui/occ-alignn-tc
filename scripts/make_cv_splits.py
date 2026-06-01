"""Create grouped cross-validation CSVs for Tc training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from occ_alignn.analysis.regimes import add_regime_columns


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


def _group_metadata(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    enriched = add_regime_columns(df)
    groups = _group_series(enriched, group_column)
    meta = enriched.assign(__group=groups).groupby("__group", dropna=False).agg(
        n=("__group", "size"),
        family=("family", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        tc_bin=("tc_bin", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        pressure_bin=("pressure_bin", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        regime=("regime", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
    )
    return meta.reset_index().rename(columns={"__group": "group"})


def make_stratified_group_split(
    df: pd.DataFrame,
    group_column: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Create one parent-group-preserving split balanced by coarse regimes."""
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must leave a positive test fraction.")
    groups = _group_series(df, group_column)
    meta = _group_metadata(df, group_column)
    rng = np.random.default_rng(seed)
    split_by_group: dict[str, str] = {}
    target = {"train": train_frac, "val": val_frac}
    strata = meta.groupby(["family", "tc_bin", "pressure_bin"], dropna=False, observed=False)
    for _, stratum in strata:
        group_names = stratum["group"].astype(str).to_numpy()
        rng.shuffle(group_names)
        n = len(group_names)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0
        for group in group_names[:n_train]:
            split_by_group[str(group)] = "train"
        for group in group_names[n_train : n_train + n_val]:
            split_by_group[str(group)] = "val"
        for group in group_names[n_train + n_val :]:
            split_by_group[str(group)] = "test"
    out = df.copy()
    out["split"] = groups.astype(str).map(split_by_group).fillna("train").to_numpy()
    return out


def make_holdout_split(
    df: pd.DataFrame,
    group_column: str,
    holdout_column: str,
    holdout_value: str,
    val_group_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Create a split where any group containing holdout_value goes to test."""
    enriched = add_regime_columns(df)
    if holdout_column not in enriched.columns:
        raise ValueError(f"holdout_column {holdout_column!r} is not present.")
    groups = _group_series(enriched, group_column)
    holdout_mask = enriched[holdout_column].astype(str) == str(holdout_value)
    test_groups = set(groups[holdout_mask].astype(str))
    all_groups = sorted(set(groups.astype(str)))
    remaining_groups = [group for group in all_groups if group not in test_groups]
    rng = np.random.default_rng(seed)
    shuffled_remaining = np.asarray(remaining_groups)
    rng.shuffle(shuffled_remaining)
    n_val = max(1, int(round(len(shuffled_remaining) * val_group_frac))) if len(shuffled_remaining) else 0
    val_groups = set(shuffled_remaining[:n_val])
    out = df.copy()
    out["split"] = [
        "test" if str(group) in test_groups else "val" if str(group) in val_groups else "train"
        for group in groups
    ]
    return out


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
        if "family" in df.columns:
            row[f"{split}_families"] = int(df.loc[mask, "family"].nunique())
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
    parser.add_argument(
        "--mode",
        choices=["grouped_cv", "stratified", "family_holdout", "regime_holdout"],
        default="grouped_cv",
    )
    parser.add_argument("--group_column", default="parent_cif_id")
    parser.add_argument("--val_group_frac", type=float, default=0.1)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--holdout_column", default=None)
    parser.add_argument("--holdout_value", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    df = pd.read_csv(data_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "grouped_cv":
        fold_frames = make_cv_splits(
            df,
            n_folds=args.n_folds,
            group_column=args.group_column,
            val_group_frac=args.val_group_frac,
            seed=args.seed,
        )
    elif args.mode == "stratified":
        fold_frames = [
            make_stratified_group_split(
                df,
                group_column=args.group_column,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                seed=args.seed,
            )
        ]
    else:
        holdout_column = args.holdout_column or ("family" if args.mode == "family_holdout" else "regime")
        if not args.holdout_value:
            raise ValueError(f"--holdout_value is required for mode {args.mode}.")
        fold_frames = [
            make_holdout_split(
                df,
                group_column=args.group_column,
                holdout_column=holdout_column,
                holdout_value=args.holdout_value,
                val_group_frac=args.val_group_frac,
                seed=args.seed,
            )
        ]
    summary = []
    for fold, fold_df in enumerate(fold_frames):
        if args.mode != "grouped_cv":
            fold_df["cv_fold"] = fold
        fold_df = _rewrite_relative_paths(fold_df, data_csv.parent, out_dir)
        filename = f"fold_{fold}.csv" if args.mode == "grouped_cv" else f"{args.mode}.csv"
        fold_df.to_csv(out_dir / filename, index=False)
        summary.append(_summary_row(fold_df, fold=fold, group_column=args.group_column))
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
