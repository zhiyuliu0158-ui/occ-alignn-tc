"""Summarize Tc prediction runs and error diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from occ_alignn.training.metrics import grouped_metrics, regression_metrics
from occ_alignn.utils.io import ensure_dir


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def _prediction_path(run_dir: Path, split: str) -> Path:
    return run_dir / f"predictions_{split}.csv"


def _metrics_row(label: str, split: str, frame: pd.DataFrame) -> dict[str, object]:
    metrics = regression_metrics(frame["Tc_true_K"].to_numpy(), frame["Tc_pred_K"].to_numpy())
    return {"run": label, "split": split, "n": len(frame), **metrics}


def _subset_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    pressure = pd.to_numeric(frame.get("pressure_GPa", 0.0), errors="coerce").fillna(0.0)
    formula = frame["formula_standardized"].astype(str) if "formula_standardized" in frame else pd.Series("", index=frame.index)
    match_type = frame["match_type"].astype(str) if "match_type" in frame else pd.Series("", index=frame.index)
    return {
        "all": pd.Series(True, index=frame.index),
        "exclude_H3S": formula != "H3S",
        "exclude_Tc_gt_120": frame["Tc_true_K"] <= 120.0,
        "high_pressure_gt_50": pressure > 50.0,
        "formula_exact": match_type == "formula_exact",
        "formula_similarity": match_type == "formula_similarity",
    }


def _add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pressure = pd.to_numeric(out.get("pressure_GPa", 0.0), errors="coerce").fillna(0.0)
    field = pd.to_numeric(out.get("magnetic_field_T", 0.0), errors="coerce").fillna(0.0)
    out["tc_bin"] = pd.cut(
        out["Tc_true_K"],
        [-0.1, 1, 5, 10, 20, 40, 80, 120, float("inf")],
        labels=["<=1", "1-5", "5-10", "10-20", "20-40", "40-80", "80-120", ">120"],
    )
    out["pressure_bin"] = pd.cut(
        pressure,
        [-0.1, 0, 1, 10, 50, 100, float("inf")],
        labels=["0", "0-1", "1-10", "10-50", "50-100", ">100"],
    )
    out["field_bin"] = pd.cut(
        field,
        [-0.1, 0, 0.01, 1, 10, float("inf")],
        labels=["0", "0-0.01", "0.01-1", "1-10", ">10"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Label=run_dir or run_dir.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--worst_n", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    comparison_rows: list[dict[str, object]] = []
    subset_rows: list[dict[str, object]] = []
    group_frames: list[pd.DataFrame] = []
    worst_frames: list[pd.DataFrame] = []

    for run_value in args.run:
        label, run_dir = _parse_run(run_value)
        for split in args.splits:
            path = _prediction_path(run_dir, split)
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            comparison_rows.append(_metrics_row(label, split, frame))
            for subset_name, mask in _subset_masks(frame).items():
                subset = frame[mask.fillna(False)]
                subset_rows.append(
                    {
                        "run": label,
                        "split": split,
                        "subset": subset_name,
                        "n": len(subset),
                        **regression_metrics(subset["Tc_true_K"].to_numpy(), subset["Tc_pred_K"].to_numpy()),
                    }
                )
            binned = _add_bins(frame)
            groups = grouped_metrics(
                binned,
                [
                    "family",
                    "match_type",
                    "fidelity",
                    "structure_source",
                    "tc_bin",
                    "pressure_bin",
                    "field_bin",
                ],
            )
            if not groups.empty:
                groups["diagnostic_split"] = split
                groups["run"] = label
                group_frames.append(groups)
            worst = frame.assign(abs_error_K=(frame["Tc_pred_K"] - frame["Tc_true_K"]).abs())
            worst = worst.sort_values("abs_error_K", ascending=False).head(args.worst_n)
            worst["diagnostic_split"] = split
            worst["run"] = label
            worst_frames.append(worst)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "comparison_metrics.csv", index=False)
    if not comparison.empty:
        try:
            markdown = comparison.to_markdown(index=False)
        except Exception:
            markdown = comparison.to_csv(index=False)
        (out_dir / "comparison_metrics.md").write_text(markdown, encoding="utf-8")
    pd.DataFrame(subset_rows).to_csv(out_dir / "subset_metrics.csv", index=False)
    if group_frames:
        pd.concat(group_frames, ignore_index=True).to_csv(out_dir / "grouped_metrics.csv", index=False)
    if worst_frames:
        pd.concat(worst_frames, ignore_index=True).to_csv(out_dir / "worst_errors.csv", index=False)


if __name__ == "__main__":
    main()
