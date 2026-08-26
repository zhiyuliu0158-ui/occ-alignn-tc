#!/usr/bin/env python
"""Adapt the cleaned SC-Atlas public table to the current Tc model schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FILM_LIKE_SAMPLE_FORMS = {"thin_film", "monolayer", "interface"}
MISSING_TEXT = {"", "nan", "none", "null", "unknown", "not reported", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def clean_optional_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(MISSING_TEXT), pd.NA)


def measurement_method(series: pd.Series) -> pd.Series:
    values = series.fillna("unknown").astype(str).str.strip().str.lower()
    mapping = {
        "onset": "onset",
        "midpoint": "midpoint",
        "zero": "zero",
        "zero resistance": "zero",
    }
    return values.map(mapping).fillna("unknown")


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.input_csv, low_memory=False)
    required = {
        "source_record_id",
        "formula_normalized",
        "tc_value_k",
        "tc_criterion",
        "pressure_gpa",
        "magnetic_field_t",
        "sample_form",
        "preparation_method",
        "substrate",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"Input table is missing required columns: {missing}")

    record_id = source["source_record_id"].astype(str).str.strip()
    if record_id.eq("").any() or record_id.duplicated().any():
        raise RuntimeError("source_record_id must be nonempty and unique")

    formula = source["formula_normalized"].astype(str).str.strip()
    tc = pd.to_numeric(source["tc_value_k"], errors="coerce")
    if formula.eq("").any() or tc.isna().any() or (~np.isfinite(tc)).any():
        raise RuntimeError("formula_normalized and tc_value_k must be complete and finite")

    sample_form = source["sample_form"].fillna("unknown").astype(str).str.strip().str.lower()
    preparation = source["preparation_method"].fillna("unknown").astype(str).str.strip()
    film_like = sample_form.isin(FILM_LIKE_SAMPLE_FORMS)
    # The existing physical-substrate encoder infers applicability from synthesis text.
    # Add sample-form evidence only for that status flag; synthesis is not one-hot encoded.
    synthesis = preparation.copy()
    synthesis.loc[film_like] = synthesis.loc[film_like] + "; thin film sample form"

    adapted = pd.DataFrame(
        {
            "record_id": record_id,
            "formula_standardized": formula,
            "formula_reduced": formula,
            "tc_k_standardized": tc.astype(float),
            "pressure_gpa_standardized": pd.to_numeric(source["pressure_gpa"], errors="coerce"),
            "pressure_raw": pd.NA,
            "magnetic_field_t_standardized": pd.to_numeric(
                source["magnetic_field_t"], errors="coerce"
            ),
            "magnetic_field_raw": pd.NA,
            "tc_measurement_method": measurement_method(source["tc_criterion"]),
            "criterion_raw": source["tc_criterion"].fillna("unknown").astype(str).str.strip(),
            "synthesis_method": synthesis,
            "substrate": clean_optional_text(source["substrate"]),
            "sample_form": sample_form,
            "fixed_split": "unassigned",
            "primary_cif_relpath": "",
        }
    )
    for column in ["doi", "source_title", "source_year", "formula_raw"]:
        if column in source.columns:
            adapted[column] = source[column]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    pressure = adapted["pressure_gpa_standardized"]
    high_tc = adapted["tc_k_standardized"].gt(200.0)
    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows": int(len(adapted)),
        "unique_record_ids": int(adapted["record_id"].nunique()),
        "unique_formula_strings": int(adapted["formula_standardized"].nunique()),
        "pressure_missing_rows": int(adapted["pressure_gpa_standardized"].isna().sum()),
        "magnetic_field_missing_rows": int(
            adapted["magnetic_field_t_standardized"].isna().sum()
        ),
        "known_substrate_rows": int(adapted["substrate"].notna().sum()),
        "film_like_sample_form_rows": int(film_like.sum()),
        "tc_above_200_rows": int(high_tc.sum()),
        "tc_above_200_without_positive_pressure": int((high_tc & ~pressure.gt(0)).sum()),
        "measurement_method_counts": adapted["tc_measurement_method"].value_counts().to_dict(),
        "sample_form_counts": adapted["sample_form"].value_counts().to_dict(),
        "policy": {
            "missing_numeric_conditions": "preserved as missing",
            "sample_form": "used only as film-like evidence for physical substrate status",
            "cif_features": "not used",
        },
    }
    args.output_csv.with_name("adapter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
