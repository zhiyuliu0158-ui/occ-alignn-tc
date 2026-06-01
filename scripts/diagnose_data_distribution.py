"""Diagnose Tc dataset split, family, pressure, Tc, and CIF reuse distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from occ_alignn.analysis.regimes import add_regime_columns, numeric_column
from occ_alignn.utils.io import ensure_dir


def _mode_or_first(values: pd.Series) -> object:
    clean = values.dropna()
    if clean.empty:
        return ""
    mode = clean.mode()
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _summary_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_col in group_cols:
        if group_col not in frame.columns:
            continue
        for value, group in frame.groupby(group_col, dropna=False, observed=False):
            tc = numeric_column(group, "Tc_K", default=float("nan"))
            pressure = numeric_column(group, "pressure_GPa", default=0.0)
            rows.append(
                {
                    "group_column": group_col,
                    "group_value": value,
                    "n": len(group),
                    "formulas": group["formula_standardized"].nunique()
                    if "formula_standardized" in group
                    else 0,
                    "parent_cifs": group["parent_cif_id"].nunique() if "parent_cif_id" in group else 0,
                    "tc_mean": tc.mean(),
                    "tc_median": tc.median(),
                    "tc_max": tc.max(),
                    "p_gt_0": int((pressure > 0).sum()),
                    "p_gt_50": int((pressure > 50).sum()),
                    "p_gt_100": int((pressure > 100).sum()),
                    "tc_gt_40": int((tc > 40).sum()),
                    "tc_gt_90": int((tc > 90).sum()),
                    "tc_gt_120": int((tc > 120).sum()),
                }
            )
    return pd.DataFrame(rows)


def _cif_reuse(frame: pd.DataFrame) -> pd.DataFrame:
    if "parent_cif_id" not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby("parent_cif_id", dropna=False)
    out = grouped.agg(
        n=("parent_cif_id", "size"),
        formulas=("formula_standardized", "nunique") if "formula_standardized" in frame else ("parent_cif_id", "size"),
        tc_min=("Tc_K", "min"),
        tc_median=("Tc_K", "median"),
        tc_max=("Tc_K", "max"),
        p_min=("pressure_GPa", "min") if "pressure_GPa" in frame else ("parent_cif_id", "size"),
        p_max=("pressure_GPa", "max") if "pressure_GPa" in frame else ("parent_cif_id", "size"),
        family=("family", _mode_or_first) if "family" in frame else ("parent_cif_id", _mode_or_first),
    ).reset_index()
    out["tc_span"] = out["tc_max"] - out["tc_min"]
    out["p_span"] = out["p_max"] - out["p_min"]
    return out.sort_values(["n", "tc_span"], ascending=False)


def _formula_cif_span(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"formula_standardized", "parent_cif_id", "Tc_K"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = (
        frame.groupby(["formula_standardized", "parent_cif_id"], dropna=False)
        .agg(
            n=("Tc_K", "size"),
            tc_min=("Tc_K", "min"),
            tc_median=("Tc_K", "median"),
            tc_max=("Tc_K", "max"),
            p_min=("pressure_GPa", "min") if "pressure_GPa" in frame else ("Tc_K", "size"),
            p_max=("pressure_GPa", "max") if "pressure_GPa" in frame else ("Tc_K", "size"),
            family=("family", _mode_or_first) if "family" in frame else ("Tc_K", _mode_or_first),
        )
        .reset_index()
    )
    out["tc_span"] = out["tc_max"] - out["tc_min"]
    out["p_span"] = out["p_max"] - out["p_min"]
    return out.sort_values(["tc_span", "n"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    frame = pd.read_csv(args.data_csv)
    for numeric_name in ("Tc_K", "pressure_GPa", "magnetic_field_T"):
        if numeric_name in frame.columns:
            frame[numeric_name] = pd.to_numeric(frame[numeric_name], errors="coerce")
    frame = add_regime_columns(frame)
    frame.to_csv(out_dir / "data_with_regimes.csv", index=False)

    summary = _summary_by(
        frame,
        [
            args.split_column,
            "family",
            "match_type",
            "fidelity",
            "structure_source",
            "tc_bin",
            "pressure_bin",
            "field_bin",
            "regime",
        ],
    )
    summary.to_csv(out_dir / "distribution_summary.csv", index=False)

    if args.split_column in frame.columns:
        for column in ["family", "tc_bin", "pressure_bin", "regime", "match_type", "fidelity"]:
            if column in frame.columns:
                pd.crosstab(frame[column], frame[args.split_column]).to_csv(out_dir / f"{column}_by_split.csv")

    reuse = _cif_reuse(frame)
    if not reuse.empty:
        reuse.to_csv(out_dir / "parent_cif_reuse.csv", index=False)
        reuse.head(args.top_n).to_csv(out_dir / "parent_cif_reuse_top.csv", index=False)

    formula_span = _formula_cif_span(frame)
    if not formula_span.empty:
        formula_span.to_csv(out_dir / "formula_parent_cif_span.csv", index=False)
        formula_span.head(args.top_n).to_csv(out_dir / "formula_parent_cif_span_top.csv", index=False)

    high_pressure = frame[numeric_column(frame, "pressure_GPa", default=0.0) > 50.0]
    high_pressure.to_csv(out_dir / "high_pressure_rows.csv", index=False)
    if not high_pressure.empty:
        _summary_by(high_pressure, ["formula_standardized", "parent_cif_id", "family", "regime"]).to_csv(
            out_dir / "high_pressure_concentration.csv",
            index=False,
        )

    print(f"wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
