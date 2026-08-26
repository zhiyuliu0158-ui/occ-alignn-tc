#!/usr/bin/env python
"""Remove high-pressure hydride families from the cleaned Tc dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "md_only_cif_3dsc" / "scripts").exists():
            return parent
    return here.parents[2]


ROOT = workspace_root()
SCRIPTS_DIR = ROOT / "md_only_cif_3dsc" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_mp_exact_cifs import parse_formula  # noqa: E402
from evaluate_final_cleanned_3dsc_official import (  # noqa: E402
    chemical_system,
    clean_formula_for_magpie,
)


# These elements indicate molecular, oxide, nitride, hydroxide, or halide
# frameworks rather than the binary/ternary hydride families targeted here.
NON_HYDRIDE_FRAMEWORK_ELEMENTS = {"C", "N", "O", "F", "Cl", "Br", "I"}


def parse_args() -> argparse.Namespace:
    base_dir = ROOT / "md_only_cif_3dsc"
    audit_dir = base_dir / "formula_sample_quality_audit_20260805"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=audit_dir / "corrected_13043_drop_high_confidence_115.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "high_pressure_hydride_ablation_20260805",
    )
    parser.add_argument("--pressure-threshold-gpa", type=float, default=5.0)
    parser.add_argument(
        "--superhydride-h-ratio",
        type=float,
        default=2.0,
        help=(
            "For missing-pressure rows only, infer a superhydride when H atoms per "
            "non-H atom reach this ratio. Rows explicitly recorded at zero pressure "
            "are never inferred this way."
        ),
    )
    return parser.parse_args()


def parse_raw_pressure_gpa(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if "ambient" in text:
        return 0.0
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan
    if "gpa" in text:
        factor = 1.0
    elif "mbar" in text:
        factor = 100.0
    elif "kbar" in text:
        factor = 0.1
    elif re.search(r"\bbar\b", text):
        factor = 1.0e-4
    elif "mpa" in text:
        factor = 1.0e-3
    elif "kpa" in text:
        factor = 1.0e-6
    elif re.search(r"\bpa\b", text):
        factor = 1.0e-9
    else:
        return np.nan
    return max(numbers) * factor


def composition_stats(formula: str) -> tuple[set[str], float]:
    composition = parse_formula(formula)
    amounts: dict[str, float] = {}
    for element, amount in composition.items():
        symbol = "H" if str(element) in {"D", "T"} else str(element)
        amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
    elements = set(amounts)
    hydrogen = amounts.get("H", 0.0)
    non_hydrogen = sum(amount for symbol, amount in amounts.items() if symbol != "H")
    ratio = hydrogen / non_hydrogen if non_hydrogen > 0 else np.inf
    return elements, ratio


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv, low_memory=False)
    if data["record_id"].duplicated().any():
        raise RuntimeError("Input contains duplicate record_id values.")

    data["formula_sc_audit"] = data.apply(clean_formula_for_magpie, axis=1)
    data["chemical_system_audit"] = data["formula_sc_audit"].map(chemical_system)
    stats = data["formula_sc_audit"].map(composition_stats)
    data["elements_audit"] = stats.map(lambda item: "-".join(sorted(item[0])))
    data["hydrogen_to_nonhydrogen_ratio"] = stats.map(lambda item: item[1])
    element_sets = stats.map(lambda item: item[0])
    data["hydride_like_composition"] = element_sets.map(
        lambda elements: "H" in elements
        and not bool(elements & NON_HYDRIDE_FRAMEWORK_ELEMENTS)
    )

    standardized_pressure = pd.to_numeric(
        data["pressure_gpa_standardized"], errors="coerce"
    )
    raw_pressure = data["pressure_raw"].map(parse_raw_pressure_gpa)
    data["pressure_evidence_gpa"] = standardized_pressure.where(
        standardized_pressure.notna(), raw_pressure
    )
    data["pressure_evidence_source"] = np.select(
        [standardized_pressure.notna(), raw_pressure.notna()],
        ["pressure_gpa_standardized", "pressure_raw_unit_converted"],
        default="missing",
    )

    data["direct_high_pressure_hydride"] = data["hydride_like_composition"] & (
        data["pressure_evidence_gpa"] > args.pressure_threshold_gpa
    )
    data["inferred_missing_pressure_superhydride"] = (
        data["hydride_like_composition"]
        & data["pressure_evidence_gpa"].isna()
        & (
            data["hydrogen_to_nonhydrogen_ratio"]
            >= args.superhydride_h_ratio
        )
    )

    seed = data[
        data["direct_high_pressure_hydride"]
        | data["inferred_missing_pressure_superhydride"]
    ]
    high_pressure_systems = set(seed["chemical_system_audit"])
    data["remove_high_pressure_hydride_family"] = (
        data["hydride_like_composition"]
        & data["chemical_system_audit"].isin(high_pressure_systems)
    )
    data["removal_reason"] = np.select(
        [
            data["direct_high_pressure_hydride"],
            data["inferred_missing_pressure_superhydride"],
            data["remove_high_pressure_hydride_family"],
        ],
        [
            "direct_pressure_above_threshold",
            "missing_pressure_superhydride_stoichiometry",
            "same_hydride_chemical_system_family",
        ],
        default="retained",
    )

    removed = data.loc[data["remove_high_pressure_hydride_family"]].copy()
    retained = data.loc[~data["remove_high_pressure_hydride_family"]].copy()
    audit_columns = [
        "formula_sc_audit",
        "chemical_system_audit",
        "elements_audit",
        "hydrogen_to_nonhydrogen_ratio",
        "hydride_like_composition",
        "pressure_evidence_gpa",
        "pressure_evidence_source",
        "direct_high_pressure_hydride",
        "inferred_missing_pressure_superhydride",
        "remove_high_pressure_hydride_family",
        "removal_reason",
    ]
    retained = retained.drop(columns=audit_columns)

    retained_path = output_dir / "corrected_drop115_drop_high_pressure_hydrides.csv"
    removed_path = output_dir / "removed_high_pressure_hydride_records.csv"
    systems_path = output_dir / "removed_high_pressure_hydride_systems.csv"
    retained.to_csv(retained_path, index=False)
    removed.to_csv(removed_path, index=False)

    systems = (
        removed.groupby("chemical_system_audit", as_index=False)
        .agg(
            removed_rows=("record_id", "size"),
            unique_formulas=("formula_sc_audit", "nunique"),
            direct_high_pressure_rows=("direct_high_pressure_hydride", "sum"),
            inferred_missing_pressure_rows=(
                "inferred_missing_pressure_superhydride",
                "sum",
            ),
            tc_min_k=("tc_k_standardized", "min"),
            tc_max_k=("tc_k_standardized", "max"),
            pressure_evidence_max_gpa=("pressure_evidence_gpa", "max"),
        )
        .sort_values(["removed_rows", "chemical_system_audit"], ascending=[False, True])
    )
    systems.to_csv(systems_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "retained_csv": str(retained_path),
        "removed_records_csv": str(removed_path),
        "pressure_threshold_gpa": args.pressure_threshold_gpa,
        "superhydride_h_to_nonhydrogen_ratio": args.superhydride_h_ratio,
        "input_rows": int(len(data)),
        "removed_rows": int(len(removed)),
        "retained_rows": int(len(retained)),
        "removed_unique_formula_sc": int(removed["formula_sc_audit"].nunique()),
        "removed_chemical_systems": int(len(high_pressure_systems)),
        "direct_high_pressure_rows": int(
            removed["direct_high_pressure_hydride"].sum()
        ),
        "inferred_missing_pressure_superhydride_rows": int(
            removed["inferred_missing_pressure_superhydride"].sum()
        ),
        "same_system_family_expansion_rows": int(
            (
                ~removed["direct_high_pressure_hydride"]
                & ~removed["inferred_missing_pressure_superhydride"]
            ).sum()
        ),
        "blocked_framework_elements": sorted(NON_HYDRIDE_FRAMEWORK_ELEMENTS),
    }
    (output_dir / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# High-pressure hydride ablation",
        "",
        f"- Input records: {len(data):,}",
        f"- Removed records: {len(removed):,}",
        f"- Retained records: {len(retained):,}",
        f"- Removed chemical systems: {len(high_pressure_systems):,}",
        f"- Direct pressure evidence above {args.pressure_threshold_gpa:g} GPa: "
        f"{int(removed['direct_high_pressure_hydride'].sum()):,}",
        "- Missing-pressure superhydrides inferred from stoichiometry: "
        f"{int(removed['inferred_missing_pressure_superhydride'].sum()):,}",
        "- Additional same-system hydride-family records: "
        f"{int((~removed['direct_high_pressure_hydride'] & ~removed['inferred_missing_pressure_superhydride']).sum()):,}",
        "",
        "The family rule excludes compositions containing C, N, O, or halogens so that",
        "organic superconductors, hydroxides, oxides, nitrides, and hydrogen-intercalated",
        "framework compounds are not mislabeled as conventional high-pressure hydrides.",
        "Rows explicitly recorded at zero pressure are not inferred as superhydrides solely",
        "from stoichiometry, but hydrides in a chemical system with direct high-pressure",
        "evidence are removed as a complete family for the ablation test.",
    ]
    (output_dir / "filter_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nRemoved systems:\n" + systems.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
