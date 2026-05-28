"""Summarize cross-validation fold metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv_root", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--pattern", default="fold_*")
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    rows: list[dict[str, object]] = []
    for fold_dir in sorted(cv_root.glob(args.pattern)):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            fold = fold_dir.name
        for split in ("val", "test"):
            metrics_path = fold_dir / f"metrics_{split}.json"
            if metrics_path.exists():
                rows.append({"fold": fold, "split": split, **_read_metrics(metrics_path)})
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError(f"No fold metrics found under {cv_root}.")
    summary = metrics.groupby("split").agg(["mean", "std"])
    out_csv = Path(args.out_csv) if args.out_csv else cv_root / "cv_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    summary.to_csv(cv_root / "cv_metrics_summary.csv")
    print(metrics.to_string(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()
