#!/usr/bin/env python
"""Run 3DSC-style XGBoost Tc models with optional ALIGNN-DOS eDOS features.

This is an adapter for the current md_only_cif_3dsc package.  It reuses the
official-style 3DSC model pieces already present in this repository, but reads
the current 01_exact_cif_only train/test split and the eDOS features generated
from ALIGNN-DOS CIF predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "md_only_cif_3dsc" / "scripts" / "evaluate_3dsc_tc.py").exists():
            return parent
    return here.parents[2]


def add_import_paths() -> None:
    root = workspace_root()
    scripts_dir = root / "md_only_cif_3dsc" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


add_import_paths()

from download_mp_exact_cifs import FormulaError, parse_formula, write_csv_xlsx  # noqa: E402
from evaluate_final_cleanned_3dsc_official import (  # noqa: E402
    STRUCTURE_AUX_FEATURES,
    calculate_magpie_features,
    calculate_structure_features,
    chemical_system,
    clean_formula_for_magpie,
    composition_to_formula,
    metrics_3dsc,
    restricted_arcsinh,
    restricted_sinh,
    resolve_cif_paths,
)


SUMMARY_EDOS_COLUMNS = [
    "edos_available",
    "edos_sum",
    "edos_mean",
    "edos_std",
    "edos_max",
    "edos_min",
    "edos_fermi_dos",
    "edos_peak_energy_ev",
    "edos_peak_value",
    "edos_centroid_ev",
    "edos_rms_energy_ev",
    "edos_width_ev",
    "edos_skewness",
    "edos_kurtosis",
    "edos_entropy",
    "edos_q10_ev",
    "edos_q25_ev",
    "edos_q50_ev",
    "edos_q75_ev",
    "edos_q90_ev",
    "edos_occupied_frac",
    "edos_unoccupied_frac",
    "edos_valence_deep_m5_m2_frac",
    "edos_valence_m2_m0p5_frac",
    "edos_valence_near_m0p5_0_frac",
    "edos_conduction_near_0_0p5_frac",
    "edos_conduction_0p5_2_frac",
    "edos_conduction_high_2_10_frac",
    "edos_near_fermi_pm_0p1_frac",
    "edos_near_fermi_pm_0p2_frac",
    "edos_near_fermi_pm_0p5_frac",
    "edos_near_fermi_pm_1p0_frac",
    "edos_gap_ev_abs_0p02",
    "edos_gap_ev_rel_2pct",
    "edos_is_metal_abs_0p02",
    "edos_is_metal_rel_2pct",
    "edos_support_min_ev",
    "edos_support_max_ev",
    "edos_support_width_ev",
    "edos_n_atoms",
    "edos_prediction_is_mp_like",
]

NEAR_FERMI_EDOS_COLUMNS = [
    "edos_fermi_dos",
    "edos_valence_near_m0p5_0_frac",
    "edos_conduction_near_0_0p5_frac",
    "edos_near_fermi_pm_0p1_frac",
    "edos_near_fermi_pm_0p2_frac",
    "edos_near_fermi_pm_0p5_frac",
    "edos_near_fermi_pm_1p0_frac",
    "edos_gap_ev_abs_0p02",
    "edos_gap_ev_rel_2pct",
    "edos_is_metal_abs_0p02",
    "edos_is_metal_rel_2pct",
]

BAND_EDOS_COLUMNS = [
    "edos_occupied_frac",
    "edos_unoccupied_frac",
    "edos_valence_deep_m5_m2_frac",
    "edos_valence_m2_m0p5_frac",
    "edos_valence_near_m0p5_0_frac",
    "edos_conduction_near_0_0p5_frac",
    "edos_conduction_0p5_2_frac",
    "edos_conduction_high_2_10_frac",
    "edos_centroid_ev",
    "edos_rms_energy_ev",
    "edos_width_ev",
    "edos_q10_ev",
    "edos_q25_ev",
    "edos_q50_ev",
    "edos_q75_ev",
    "edos_q90_ev",
]

FERMI_DERIVED_EDOS_COLUMNS = [
    "edos_fermi_slope",
    "edos_fermi_curvature",
    "edos_fermi_asymmetry_pm_0p5",
    "edos_fermi_asymmetry_pm_1p0",
    "edos_fermi_to_mean_ratio",
    "edos_near_fermi_peak_value_pm_0p5",
    "edos_near_fermi_peak_energy_pm_0p5",
    "edos_near_fermi_min_value_pm_0p5",
    "edos_near_fermi_peak_to_fermi_ratio_pm_0p5",
]

SUMMARY_PHDOS_COLUMNS = [
    "phdos_available",
    "phdos_sum",
    "phdos_mean",
    "phdos_std",
    "phdos_max",
    "phdos_peak_freq_cm-1",
    "phdos_peak_value",
    "phdos_centroid_cm-1",
    "phdos_rms_freq_cm-1",
    "phdos_width_cm-1",
    "phdos_skewness",
    "phdos_kurtosis",
    "phdos_entropy",
    "phdos_q10_cm-1",
    "phdos_q25_cm-1",
    "phdos_q50_cm-1",
    "phdos_q75_cm-1",
    "phdos_q90_cm-1",
    "phdos_low_0_100_frac",
    "phdos_low_0_200_frac",
    "phdos_mid_200_500_frac",
    "phdos_high_500_1000_frac",
    "phdos_hard_gt_600_frac",
    "phdos_soft_lt_80_frac",
    "phdos_support_min_cm-1",
    "phdos_support_max_cm-1",
    "phdos_support_width_cm-1",
    "phdos_num_sites",
    "phdos_supercell_mult",
    "phdos_ordered_from_disordered",
]

PHYSICAL_PHDOS_COLUMNS = [
    "phdos_log_freq_cm-1",
    "phdos_log_temp_k",
    "phdos_ad_prefactor_k",
    "phdos_centroid_temp_k",
    "phdos_rms_temp_k",
    "phdos_q90_temp_k",
    "phdos_support_max_temp_k",
    "phdos_mean_inv_freq_cm",
    "phdos_mean_inv_freq2_cm2",
    "phdos_softness_m1_shift20",
    "phdos_softness_m2_shift20",
    "phdos_freq_variation_coeff",
    "phdos_centroid_sq_cm-2",
    "phdos_rms_sq_cm-2",
    "phdos_effective_bin_count",
    "phdos_effective_bin_fraction",
    "phdos_spectral_roughness",
    "phdos_largest_gap_cm-1",
    "phdos_largest_gap_fraction",
    "phdos_peak_count_10pct",
    "phdos_lowfreq_slope_20_120",
    "phdos_lowfreq_debye_coeff_20_120",
    "phdos_lowfreq_debye_r2_20_120",
    "phdos_frac_0_50",
    "phdos_frac_50_100",
    "phdos_frac_100_200",
    "phdos_frac_200_400",
    "phdos_frac_400_700",
    "phdos_frac_700_1000",
    "phdos_frac_0_120",
    "phdos_frac_120_300",
    "phdos_frac_300_1000",
    "phdos_acoustic_to_optical_ratio",
    "phdos_optical_to_acoustic_ratio",
    "phdos_soft_to_stiff_ratio",
    "phdos_low_to_high_ratio",
    "phdos_soft_mode_score",
    "phdos_hard_mode_score",
]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    base_dir = root / "md_only_cif_3dsc"
    parser = argparse.ArgumentParser(
        description="3DSC-style XGB Tc evaluation with ALIGNN-DOS eDOS features."
    )
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--dataset", default="01_exact_cif_only")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help=(
            "Optional input table. When supplied, it replaces the dataset's fixed "
            "train/test CSV files; group-CV can then be generated or supplied with "
            "--cv-split-file."
        ),
    )
    parser.add_argument(
        "--edos-features",
        type=Path,
        default=base_dir
        / "electronic_tc_feature_results_exact_only"
        / "edos_features_selected_by_record.csv",
    )
    parser.add_argument(
        "--phdos-features",
        type=Path,
        default=base_dir / "01_exact_cif_only" / "exact_cif_only_data_with_phdos_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "electronic_tc_xgb_3dsc_style_exact_only",
    )
    parser.add_argument(
        "--feature-sets",
        default="MAGPIE,MAGPIE+EDOS_SUMMARY,MAGPIE+EDOS_BINS,MAGPIE+EDOS_SUMMARY_BINS",
        help=(
            "Comma-separated feature sets. Supported tokens are MAGPIE, DSOAP, "
            "EDOS_SUMMARY, EDOS_NEAR_FERMI, EDOS_BANDS, EDOS_FERMI_DERIVED, EDOS_BINS, "
            "EDOS_SUMMARY_BINS, EDOS_PCA<N>, PHDOS_SUMMARY, PHDOS_BINS, "
            "PHDOS_SUMMARY_BINS, PHDOS_PHYSICAL, and PHDOS_PCA<N>."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["fixed", "group-cv"],
        default="fixed",
        help="fixed uses the package train/test split; group-cv makes 3DSC chemical-system splits.",
    )
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=58)
    parser.add_argument(
        "--cv-split-file",
        type=Path,
        default=None,
        help="Optional record_id,CV_0,... CSV used to reuse an existing group-CV split exactly.",
    )
    parser.add_argument(
        "--cif-path-mapping",
        type=Path,
        default=None,
        help=(
            "Optional record-level CSV used to override the CIF path by record_id. "
            "This supports paired original-CIF versus SQS-CIF experiments without "
            "modifying the source dataset."
        ),
    )
    parser.add_argument(
        "--cif-path-column",
        default="output_path",
        help="Path column in --cif-path-mapping (default: output_path).",
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--soap-n-jobs", type=int, default=1)
    parser.add_argument(
        "--feature-select-topk",
        type=int,
        default=0,
        help="If >0, select top-k features inside each training fold using an XGBoost importance prefit.",
    )
    parser.add_argument(
        "--skip-dsoap",
        action="store_true",
        help="Use zero SOAP vectors even if a DSOAP feature set is requested.",
    )
    parser.add_argument(
        "--include-extra-categorical",
        action="store_true",
        help="Also one-hot encode measurement method and synthesis method.",
    )
    parser.add_argument(
        "--condition-categorical-mode",
        choices=["raw", "compact", "numeric", "measurement", "substrate_only"],
        default="raw",
        help=(
            "raw one-hot encodes criterion_raw; compact maps criterion_raw into "
            "semantic flags; numeric keeps only pressure, field, and the high-pressure "
            "flag; measurement adds the standardized tc_measurement_method categories "
            "to the numeric features; substrate_only removes pressure, field, and "
            "measurement-method features while retaining the selected substrate block."
        ),
    )
    parser.add_argument(
        "--substrate-feature-mode",
        choices=[
            "raw_onehot",
            "magpie",
            "status",
            "magpie_only",
            "film_status",
            "physical",
            "physical_only",
            "interface",
            "interface_top",
            "mp_lattice",
            "physical_mp",
            "none",
        ],
        default="raw_onehot",
        help=(
            "How to encode substrate when condition features are enabled: "
            "raw_onehot reproduces the old COND.substrate_* categories; "
            "magpie uses substrate status flags plus MAGPIE descriptors of parseable substrate formulas; "
            "status keeps only substrate known/unknown/applicable flags; "
            "magpie_only keeps only substrate MAGPIE descriptors; "
            "film_status distinguishes film-with-unknown-substrate from non-film/not-applicable rows; "
            "physical adds compact substrate-family and elemental-property descriptors; "
            "physical_only keeps those compact physical descriptors without substrate-status flags; "
            "interface adds material-substrate composition contrasts; "
            "interface_top also adds one-hot buckets for the most frequent substrates; "
            "mp_lattice adds film-aware status and Materials Project lattice/symmetry features; "
            "physical_mp combines compact physical substrate descriptors with MP lattice/symmetry features; "
            "none drops substrate features."
        ),
    )
    parser.add_argument(
        "--substrate-top-n",
        type=int,
        default=10,
        help="Number of frequent known substrate identities retained by interface_top.",
    )
    parser.add_argument(
        "--substrate-mp-mapping",
        type=Path,
        default=base_dir / "substrate_mp_structures" / "substrate_mp_record_mapping.csv",
        help="Record-level Materials Project substrate structure and lattice mapping.",
    )
    parser.add_argument(
        "--exclude-unmatched-known-substrates",
        action="store_true",
        help="Drop rows with a known substrate formula but no selected MP exact-composition structure.",
    )
    parser.add_argument(
        "--exclude-condition-features",
        action="store_true",
        help="Do not include COND.* pressure/field/criterion/substrate features in any feature set.",
    )
    parser.add_argument(
        "--require-edos",
        action="store_true",
        help="Filter the dataset to records with an available eDOS prediction before training.",
    )
    parser.add_argument(
        "--require-phdos",
        action="store_true",
        help="Filter the dataset to records with an available phonon-DOS prediction before training.",
    )
    parser.add_argument(
        "--require-zero-pressure-field",
        action="store_true",
        help="Strictly filter to records with pressure_gpa_standardized == 0 and magnetic_field_t_standardized == 0.",
    )
    parser.add_argument(
        "--require-positive-pressure-above-tc-k",
        type=float,
        default=0.0,
        help=(
            "If positive, retain records above this Tc threshold only when a finite "
            "pressure_gpa_standardized > 0 is recorded."
        ),
    )
    parser.add_argument(
        "--prediction-upper-bound-k",
        type=float,
        default=200.0,
        help="Upper bound used by the inverse target transform; set above the observed Tc range.",
    )
    return parser.parse_args()


def read_current_split(dataset_root: Path) -> pd.DataFrame:
    split_root = dataset_root / "3dsc_formula_group_train_test_split"
    train_dir = split_root / "train"
    test_dir = split_root / "test"
    train_files = sorted(train_dir.glob("*_train.csv"))
    test_files = sorted(test_dir.glob("*_test.csv"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Cannot find current split under {split_root}")
    train_path = train_files[0]
    test_path = test_files[0]
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Cannot find current split under {split_root}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["fixed_split"] = "train"
    test["fixed_split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def read_input_table(dataset_root: Path, data_csv: Path | None) -> pd.DataFrame:
    if data_csv is None:
        return read_current_split(dataset_root)
    if not data_csv.exists():
        raise FileNotFoundError(f"Input data CSV does not exist: {data_csv}")
    frame = pd.read_csv(data_csv, low_memory=False)
    required = {"record_id", "formula_standardized", "tc_k_standardized"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Input data CSV is missing required columns: {missing}")
    if frame["record_id"].duplicated().any():
        raise RuntimeError("Input data CSV contains duplicate record_id values.")
    if "fixed_split" not in frame.columns:
        frame["fixed_split"] = "unassigned"
    return frame


def attach_mp_substrate_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    exclude_unmatched_known: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"MP substrate mapping does not exist: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    if "record_id" not in mapping.columns:
        raise RuntimeError("MP substrate mapping must contain record_id.")
    if mapping["record_id"].duplicated().any():
        raise RuntimeError("MP substrate mapping contains duplicate record_id values.")

    overlap = sorted((set(df.columns) & set(mapping.columns)) - {"record_id"})
    if overlap:
        mapping = mapping.drop(columns=overlap)
    merged = df.merge(mapping, on="record_id", how="left", validate="one_to_one")
    status = merged.get(
        "substrate_status", pd.Series("unknown", index=merged.index)
    ).fillna("unknown").astype(str)
    material_id = merged.get(
        "substrate_mp_material_id", pd.Series("", index=merged.index)
    ).fillna("").astype(str).str.strip()
    known = status.eq("known")
    available = material_id.ne("")
    remove = known & ~available if exclude_unmatched_known else pd.Series(False, index=merged.index)
    removed_formulas = sorted(
        merged.loc[remove, "substrate_formula_standardized"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    out = merged.loc[~remove].reset_index(drop=True)
    manifest = {
        "enabled": True,
        "mapping_path": str(mapping_path.resolve()),
        "mapping_rows": int(len(mapping)),
        "rows_before_filter": int(len(merged)),
        "rows_after_filter": int(len(out)),
        "rows_removed": int(remove.sum()),
        "known_substrate_rows_before": int(known.sum()),
        "mp_structure_available_rows_before": int((known & available).sum()),
        "exclude_unmatched_known_substrates": bool(exclude_unmatched_known),
        "removed_standardized_formulas": removed_formulas,
        "removed_standardized_formula_count": int(len(removed_formulas)),
    }
    return out, manifest


def attach_cif_path_mapping(
    df: pd.DataFrame,
    mapping_path: Path,
    path_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"CIF path mapping does not exist: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    required = {"record_id", path_column}
    missing_columns = sorted(required - set(mapping.columns))
    if missing_columns:
        raise RuntimeError(
            f"CIF path mapping is missing required columns: {missing_columns}"
        )
    if mapping["record_id"].duplicated().any():
        raise RuntimeError("CIF path mapping contains duplicate record_id values.")

    def mapped_path(value: Any) -> str:
        text = str(value).strip() if pd.notna(value) else ""
        if not text:
            return ""
        path = Path(text)
        if path.is_absolute():
            return str(path)
        candidates = [
            Path.cwd() / path,
            workspace_root() / path,
            mapping_path.parent / path,
        ]
        return str(next((item for item in candidates if item.exists()), candidates[0]))

    keep = ["record_id", path_column]
    for column in ["selection", "cif_match_type", "source_path", "output_cif"]:
        if column in mapping.columns and column not in keep:
            keep.append(column)
    mapping = mapping[keep].copy()
    rename = {path_column: "cif_override_path"}
    rename.update(
        {
            column: f"cif_mapping_{column}"
            for column in keep
            if column not in {"record_id", path_column}
        }
    )
    mapping = mapping.rename(columns=rename)
    mapping["cif_override_path"] = mapping["cif_override_path"].map(mapped_path)
    merged = df.merge(mapping, on="record_id", how="left", validate="one_to_one")
    override = merged["cif_override_path"].fillna("").astype(str).str.strip()
    exists = override.map(lambda value: bool(value) and Path(value).exists())
    manifest = {
        "enabled": True,
        "mapping_path": str(mapping_path.resolve()),
        "path_column": path_column,
        "mapping_rows": int(len(mapping)),
        "dataset_rows": int(len(df)),
        "matched_record_rows": int(merged["cif_override_path"].notna().sum()),
        "nonempty_path_rows": int(override.ne("").sum()),
        "existing_path_rows": int(exists.sum()),
        "missing_or_nonexistent_path_rows": int((~exists).sum()),
    }
    if "cif_mapping_selection" in merged.columns:
        manifest["selection_counts"] = (
            merged["cif_mapping_selection"].fillna("missing").value_counts().to_dict()
        )
    return merged, manifest


def finite_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("<empty>")
        .astype(str)
        .str.strip()
        .replace({"": "<empty>", "nan": "<empty>", "None": "<empty>"})
    )


SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        chr(0x2080 + i): str(i)
        for i in range(10)
    }
    | {
        chr(0x00B7): "",
        chr(0x2022): "",
        chr(0x2212): "-",
        chr(0x2013): "-",
        chr(0x2014): "-",
    }
)

SUBSTRATE_ALIASES = {
    "STO": "SrTiO3",
    "LAO": "LaAlO3",
    "LSAO": "LaSrAlO4",
    "LSAT": "La0.3Sr0.7Al0.65Ta0.35O3",
    "DSO": "DyScO3",
    "NGO": "NdGaO3",
    "YAO": "YAlO3",
    "YSZ": "ZrO2",
    "SIO2": "SiO2",
    "AL2O3": "Al2O3",
    "MGO": "MgO",
    "CAF2": "CaF2",
}

SUBSTRATE_UNKNOWN_VALUES = {
    "<empty>",
    "unknown",
    "nan",
    "none",
    "not specified",
    "not reported",
    "n/a",
    "na",
    "-",
}

SUBSTRATE_NOT_APPLICABLE_VALUES = {
    "bulk",
    "single crystal",
    "polycrystal",
    "powder",
    "self",
    "free standing",
    "free-standing",
}

STRICT_FILM_METHOD_RE = re.compile(
    "|".join(
        [
            r"(?:thin\s*film)",
            r"(?:\bfilm\b)",
            r"(?:epitax)",
            r"(?:\bmbe\b)",
            r"(?:molecular beam epitaxy)",
            r"(?:physical vapor deposition)",
            r"(?:\bpvd\b)",
            r"(?:chemical vapor deposition)",
            r"(?:\bcvd\b)",
            r"(?:pulsed laser deposition)",
            r"(?:\bpld\b)",
            r"(?:sputter)",
            r"(?:evaporation)",
            r"(?:laser ablation)",
            r"(?:spin coat)",
            r"(?:spin-coat)",
            r"(?:sol[- ]?gel spin)",
            r"(?:atomic layer deposition)",
            r"(?:\bald\b)",
        ]
    ),
    re.I,
)

SUBSTRATE_MAGPIE_PHYSICAL_COLUMNS = [
    "mean_Number",
    "mean_AtomicWeight",
    "mean_MeltingT",
    "mean_CovalentRadius",
    "mean_Electronegativity",
    "mean_GSvolume_pa",
    "mean_GSbandgap",
    "mean_SpaceGroupNumber",
    "maxdiff_CovalentRadius",
    "maxdiff_Electronegativity",
    "maxdiff_GSbandgap",
    "frac_sValence",
    "frac_pValence",
    "frac_dValence",
    "frac_fValence",
]

SUBSTRATE_INTERFACE_MAGPIE_COLUMNS = [
    "mean_Number",
    "mean_AtomicWeight",
    "mean_MeltingT",
    "mean_CovalentRadius",
    "mean_Electronegativity",
    "mean_GSvolume_pa",
    "mean_GSbandgap",
    "mean_SpaceGroupNumber",
]

HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"S", "Se", "Te"}
GROUP14_SEMICONDUCTORS = {"C", "Si", "Ge"}
NONMETAL_OR_METALLOID_ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Ge",
    "As",
    "Se",
    "Br",
    "Sb",
    "Te",
    "I",
}


def normalize_substrate_token(value: Any) -> str:
    text = str(value).strip().translate(SUBSCRIPT_TRANSLATION)
    text = re.sub(r"\s+", "", text)
    return text


def substrate_status(value: Any) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in SUBSTRATE_UNKNOWN_VALUES or text == "<empty>":
        return "unknown"
    if lowered in SUBSTRATE_NOT_APPLICABLE_VALUES:
        return "not_applicable"
    return "known"


def normalize_substrate_formula(value: Any) -> str | None:
    status = substrate_status(value)
    if status != "known":
        return None

    text = normalize_substrate_token(value)
    if not text:
        return None
    alias = SUBSTRATE_ALIASES.get(text.upper())
    if alias is not None:
        text = alias

    candidates = [text]
    candidates.extend(part for part in re.split(r"[/,;|+]", text) if part)
    for candidate in candidates:
        candidate = re.sub(r"[^A-Za-z0-9.()\\[\\]-]", "", candidate)
        if not candidate:
            continue
        try:
            comp = parse_formula(candidate)
        except FormulaError:
            continue
        if comp:
            from pymatgen.core.periodic_table import Element

            if any(not Element.is_valid_symbol(str(element)) for element in comp):
                continue
            return composition_to_formula(comp)
    return None


def reset_chemml_import_state() -> None:
    for name in list(sys.modules):
        if name == "chemml" or name.startswith("chemml.chem"):
            del sys.modules[name]


def strict_film_like_methods(methods: pd.Series) -> pd.Series:
    values = clean_text_series(methods).str.lower()
    return values.map(lambda value: bool(STRICT_FILM_METHOD_RE.search(value)))


def film_aware_substrate_status(
    values: pd.Series, synthesis_values: pd.Series
) -> tuple[pd.Series, pd.Series]:
    legacy_status = values.map(substrate_status)
    method_film_like = strict_film_like_methods(synthesis_values)
    lower = values.str.lower()
    explicitly_unknown = lower.isin(
        {
            "unknown",
            "not specified",
            "not reported",
            "n/a",
            "na",
            "-",
        }
    )
    known = legacy_status.eq("known")
    applicable = known | explicitly_unknown | method_film_like
    status = pd.Series("not_applicable", index=values.index, dtype=object)
    status.loc[applicable & ~known] = "unknown"
    status.loc[known] = "known"
    return status, method_film_like


def substrate_composition_features(normalized: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for formula in normalized:
        row = {
            "COND.substrate_PHYS.num_elements": 0.0,
            "COND.substrate_PHYS.stoich_total_atoms": 0.0,
            "COND.substrate_PHYS.stoich_entropy": 0.0,
            "COND.substrate_PHYS.oxygen_fraction": 0.0,
            "COND.substrate_PHYS.halogen_fraction": 0.0,
            "COND.substrate_PHYS.nitrogen_fraction": 0.0,
            "COND.substrate_PHYS.carbon_fraction": 0.0,
            "COND.substrate_PHYS.is_elemental": 0.0,
            "COND.substrate_PHYS.is_oxide": 0.0,
            "COND.substrate_PHYS.is_halide": 0.0,
            "COND.substrate_PHYS.is_nitride": 0.0,
            "COND.substrate_PHYS.is_carbide": 0.0,
            "COND.substrate_PHYS.is_chalcogenide": 0.0,
            "COND.substrate_PHYS.is_perovskite_abo3_like": 0.0,
            "COND.substrate_PHYS.is_elemental_metal": 0.0,
            "COND.substrate_PHYS.is_elemental_group14": 0.0,
        }
        if formula is None:
            rows.append(row)
            continue
        try:
            comp = parse_formula(str(formula))
        except FormulaError:
            rows.append(row)
            continue
        positive = {str(el): float(amount) for el, amount in comp.items() if float(amount) > 0}
        total = float(sum(positive.values()))
        if total <= 0:
            rows.append(row)
            continue
        elements = set(positive)
        fractions = {el: amount / total for el, amount in positive.items()}
        oxygen = fractions.get("O", 0.0)
        halogen = sum(fractions.get(el, 0.0) for el in HALOGENS)
        non_oxygen_elements = elements - {"O"}
        oxygen_amount = positive.get("O", 0.0)
        non_oxygen_amount = total - oxygen_amount
        row.update(
            {
                "COND.substrate_PHYS.num_elements": float(len(elements)),
                "COND.substrate_PHYS.stoich_total_atoms": total,
                "COND.substrate_PHYS.stoich_entropy": float(
                    -sum(frac * math.log(frac) for frac in fractions.values() if frac > 0)
                ),
                "COND.substrate_PHYS.oxygen_fraction": oxygen,
                "COND.substrate_PHYS.halogen_fraction": halogen,
                "COND.substrate_PHYS.nitrogen_fraction": fractions.get("N", 0.0),
                "COND.substrate_PHYS.carbon_fraction": fractions.get("C", 0.0),
                "COND.substrate_PHYS.is_elemental": float(len(elements) == 1),
                "COND.substrate_PHYS.is_oxide": float("O" in elements and len(elements) > 1),
                "COND.substrate_PHYS.is_halide": float(bool(elements & HALOGENS)),
                "COND.substrate_PHYS.is_nitride": float("N" in elements and len(elements) > 1),
                "COND.substrate_PHYS.is_carbide": float("C" in elements and len(elements) > 1),
                "COND.substrate_PHYS.is_chalcogenide": float(
                    bool(elements & CHALCOGENS) and not bool(elements & {"O"})
                ),
                "COND.substrate_PHYS.is_perovskite_abo3_like": float(
                    "O" in elements
                    and len(non_oxygen_elements) == 2
                    and non_oxygen_amount > 0
                    and abs((oxygen_amount / non_oxygen_amount) - 1.5) <= 0.25
                ),
                "COND.substrate_PHYS.is_elemental_metal": float(
                    len(elements) == 1
                    and not elements.issubset(NONMETAL_OR_METALLOID_ELEMENTS)
                ),
                "COND.substrate_PHYS.is_elemental_group14": float(
                    len(elements) == 1 and bool(elements & GROUP14_SEMICONDUCTORS)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, index=normalized.index)


def substrate_material_overlap_features(
    normalized: pd.Series, material_formulas: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for substrate_formula, material_formula in zip(normalized, material_formulas):
        row = {
            "COND.substrate_INTERFACE.element_jaccard": 0.0,
            "COND.substrate_INTERFACE.substrate_elements_shared_fraction": 0.0,
            "COND.substrate_INTERFACE.material_elements_shared_fraction": 0.0,
            "COND.substrate_INTERFACE.shares_oxygen": 0.0,
        }
        if substrate_formula is None:
            rows.append(row)
            continue
        try:
            substrate_elements = set(parse_formula(str(substrate_formula)))
            material_elements = set(parse_formula(str(material_formula)))
        except FormulaError:
            rows.append(row)
            continue
        union = substrate_elements | material_elements
        shared = substrate_elements & material_elements
        if union:
            row["COND.substrate_INTERFACE.element_jaccard"] = len(shared) / len(union)
        if substrate_elements:
            row["COND.substrate_INTERFACE.substrate_elements_shared_fraction"] = (
                len(shared) / len(substrate_elements)
            )
        if material_elements:
            row["COND.substrate_INTERFACE.material_elements_shared_fraction"] = (
                len(shared) / len(material_elements)
            )
        row["COND.substrate_INTERFACE.shares_oxygen"] = float("O" in shared)
        rows.append(row)
    return pd.DataFrame(rows, index=normalized.index)


def make_substrate_features(
    substrate_values: pd.Series,
    cache_dir: Path,
    mode: str,
    synthesis_values: pd.Series | None = None,
    material_formulas: pd.Series | None = None,
    material_magpie: pd.DataFrame | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = clean_text_series(substrate_values)
    legacy_statuses = values.map(substrate_status)
    synthesis = (
        clean_text_series(synthesis_values)
        if synthesis_values is not None
        else pd.Series("<empty>", index=values.index)
    )
    if mode in {"film_status", "physical", "physical_only", "interface", "interface_top"}:
        statuses, method_film_like = film_aware_substrate_status(values, synthesis)
    else:
        statuses = legacy_statuses
        method_film_like = pd.Series(False, index=values.index)
    normalized = values.map(normalize_substrate_formula)

    status_out = pd.DataFrame(index=values.index)
    status_out["COND.substrate_known"] = statuses.eq("known").astype(float)
    status_out["COND.substrate_unknown"] = statuses.eq("unknown").astype(float)
    status_out["COND.substrate_not_applicable"] = statuses.eq("not_applicable").astype(float)
    status_out["COND.substrate_applicable"] = statuses.ne("not_applicable").astype(float)
    if mode in {"film_status", "physical", "interface", "interface_top"}:
        status_out["COND.substrate_is_film_like"] = method_film_like.astype(float)
    else:
        status_out["COND.substrate_is_film_like"] = statuses.eq("known").astype(float)
    status_out["COND.substrate_formula_parseable"] = normalized.notna().astype(float)
    magpie_out = pd.DataFrame(index=values.index)

    unique_formulas = pd.Series(sorted(normalized.dropna().unique()))
    parse_failures = sorted(values.loc[statuses.eq("known") & normalized.isna()].unique().tolist())
    if len(unique_formulas) > 0:
        reset_chemml_import_state()
        magpie_unique = calculate_magpie_features(
            unique_formulas, cache_dir / "substrate_magpie_by_formula.pkl"
        )
        magpie = (
            magpie_unique.set_index("formula_sc")
            .reindex(normalized.fillna("__missing__"))
            .reset_index(drop=True)
            .fillna(0.0)
        )
        magpie.columns = [
            f"COND.substrate_MAGPIE.{col.removeprefix('MAGPIE.')}" for col in magpie.columns
        ]
        magpie_out = magpie.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if mode == "status":
        out = status_out
    elif mode == "magpie_only":
        out = magpie_out
    elif mode == "magpie":
        out = status_out.join(magpie_out)
    elif mode == "film_status":
        out = status_out
    elif mode in {"physical", "physical_only", "interface", "interface_top"}:
        composition_out = substrate_composition_features(normalized)
        selected_magpie = pd.DataFrame(index=values.index)
        for prop in SUBSTRATE_MAGPIE_PHYSICAL_COLUMNS:
            source_col = f"COND.substrate_MAGPIE.{prop}"
            if source_col in magpie_out.columns:
                selected_magpie[f"COND.substrate_PHYS.MAGPIE_{prop}"] = magpie_out[source_col]
        physical_out = composition_out.join(selected_magpie)
        out = physical_out if mode == "physical_only" else status_out.join(physical_out)

        if mode in {"interface", "interface_top"}:
            interface_out = pd.DataFrame(index=values.index)
            known = statuses.eq("known").astype(float)
            if material_magpie is not None:
                for prop in SUBSTRATE_INTERFACE_MAGPIE_COLUMNS:
                    substrate_col = f"COND.substrate_MAGPIE.{prop}"
                    material_col = f"MAGPIE.{prop}"
                    if substrate_col not in magpie_out.columns or material_col not in material_magpie.columns:
                        continue
                    interface_out[f"COND.substrate_INTERFACE.absdiff_{prop}"] = (
                        (magpie_out[substrate_col] - material_magpie[material_col]).abs() * known
                    )
            if material_formulas is not None:
                interface_out = interface_out.join(
                    substrate_material_overlap_features(normalized, material_formulas)
                )
            out = out.join(interface_out)

        if mode == "interface_top":
            known_formulas = normalized.loc[statuses.eq("known")]
            top_formulas = known_formulas.value_counts().head(max(int(top_n), 0)).index.tolist()
            identity = pd.DataFrame(index=values.index)
            for formula in top_formulas:
                token = re.sub(r"[^A-Za-z0-9]+", "_", str(formula)).strip("_") or "unknown"
                identity[f"COND.substrate_ID.{token}"] = normalized.eq(formula).astype(float)
            identity["COND.substrate_ID.other_known"] = (
                statuses.eq("known") & ~normalized.isin(top_formulas)
            ).astype(float)
            out = out.join(identity)
    else:
        raise ValueError(f"Unsupported substrate feature mode: {mode}")

    manifest = {
        "mode": mode,
        "raw_unique_values": int(values.nunique()),
        "known_rows": int(statuses.eq("known").sum()),
        "unknown_rows": int(statuses.eq("unknown").sum()),
        "not_applicable_rows": int(statuses.eq("not_applicable").sum()),
        "method_film_like_rows": int(method_film_like.sum()),
        "parseable_rows": int(normalized.notna().sum()),
        "parseable_unique_formulas": int(len(unique_formulas)),
        "parse_failure_unique_values": parse_failures[:50],
        "n_features": int(out.shape[1]),
        "n_status_features": int(len([c for c in status_out.columns if c in out.columns])),
        "n_magpie_features": int(len([c for c in out.columns if c.startswith("COND.substrate_MAGPIE.")])),
        "n_physical_features": int(len([c for c in out.columns if c.startswith("COND.substrate_PHYS.")])),
        "n_interface_features": int(len([c for c in out.columns if c.startswith("COND.substrate_INTERFACE.")])),
        "n_identity_features": int(len([c for c in out.columns if c.startswith("COND.substrate_ID.")])),
    }
    return out, manifest


def compact_criterion_features(series: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = clean_text_series(series)
    lower = values.str.lower()
    unknown = lower.isin(SUBSTRATE_UNKNOWN_VALUES) | lower.eq("<empty>")
    out = pd.DataFrame(index=values.index)
    out["COND.criterion_unknown"] = unknown.astype(float)
    out["COND.criterion_zero"] = lower.str.contains(
        r"zero|offset|vanishing|rho\s*=\s*0|r\s*=\s*0|resistivity\s*=\s*0",
        regex=True,
        na=False,
    ).astype(float)
    out["COND.criterion_onset"] = lower.str.contains(
        r"onset|start|appears|deviation|drop begins|begins to decrease",
        regex=True,
        na=False,
    ).astype(float)
    out["COND.criterion_midpoint"] = lower.str.contains(
        r"midpoint|mid-point|middle|half|50%|0\.5\s*r",
        regex=True,
        na=False,
    ).astype(float)
    out["COND.criterion_transition"] = lower.str.contains(
        r"transition|bkt|berezinskii|kosterlitz|thouless",
        regex=True,
        na=False,
    ).astype(float)
    out["COND.criterion_thermodynamic"] = lower.str.contains(
        r"specific heat|heat capacity|entropy|thermodynamic|meissner|shielding|diamagnetic",
        regex=True,
        na=False,
    ).astype(float)
    out["COND.criterion_magnetic"] = lower.str.contains(
        r"magnet|suscept|zfc|fc|shielding|meissner|diamagnetic",
        regex=True,
        na=False,
    ).astype(float)
    assigned = out.drop(columns=["COND.criterion_unknown"]).sum(axis=1).gt(0)
    out["COND.criterion_other"] = (~unknown & ~assigned).astype(float)
    manifest = {
        "mode": "compact",
        "raw_unique_values": int(values.nunique()),
        "n_features": int(out.shape[1]),
        "positive_counts": {col: int(out[col].sum()) for col in out.columns},
    }
    return out, manifest


def bool_numeric_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .astype(float)
    )


def make_mp_substrate_lattice_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    material_id = df.get(
        "substrate_mp_material_id", pd.Series("", index=df.index)
    ).fillna("").astype(str).str.strip()
    available = material_id.ne("")
    out = pd.DataFrame(index=df.index)
    out["COND.substrate_MP.available"] = available.astype(float)
    out["COND.substrate_MP.phase_review_required"] = bool_numeric_series(
        df.get(
            "substrate_mp_phase_review_required",
            pd.Series(False, index=df.index),
        )
    )
    candidate_count = finite_float_series(
        df.get(
            "substrate_mp_exact_candidate_count",
            pd.Series(0.0, index=df.index),
        )
    )
    out["COND.substrate_MP.log1p_exact_candidate_count"] = np.log1p(
        candidate_count.clip(lower=0.0)
    )
    out["COND.substrate_MP.is_stable"] = bool_numeric_series(
        df.get("substrate_mp_is_stable", pd.Series(False, index=df.index))
    )
    out["COND.substrate_MP.is_theoretical"] = bool_numeric_series(
        df.get("substrate_mp_theoretical", pd.Series(False, index=df.index))
    )
    out["COND.substrate_MP.energy_above_hull"] = finite_float_series(
        df.get("substrate_mp_energy_above_hull", pd.Series(0.0, index=df.index))
    ).clip(lower=0.0, upper=1.0)

    source_names = {
        "a": "substrate_mp_conventional_a_ang",
        "b": "substrate_mp_conventional_b_ang",
        "c": "substrate_mp_conventional_c_ang",
        "alpha": "substrate_mp_conventional_alpha_deg",
        "beta": "substrate_mp_conventional_beta_deg",
        "gamma": "substrate_mp_conventional_gamma_deg",
        "volume": "substrate_mp_conventional_volume_ang3",
        "nsites": "substrate_mp_conventional_nsites",
        "space_group_number": "substrate_mp_space_group_number_inferred",
    }
    values = {
        name: finite_float_series(
            df.get(column, pd.Series(0.0, index=df.index))
        )
        for name, column in source_names.items()
    }
    for name in ["a", "b", "c"]:
        out[f"COND.substrate_MP.conventional_{name}_ang"] = values[name]
    for name in ["alpha", "beta", "gamma"]:
        out[f"COND.substrate_MP.conventional_{name}_deg"] = values[name]
    out["COND.substrate_MP.conventional_volume_ang3"] = values["volume"]
    out["COND.substrate_MP.conventional_nsites"] = values["nsites"]
    out["COND.substrate_MP.space_group_number_norm"] = values[
        "space_group_number"
    ] / 230.0

    nonzero_a = values["a"].where(values["a"].abs() > 1e-12, np.nan)
    nonzero_sites = values["nsites"].where(values["nsites"] > 0, np.nan)
    out["COND.substrate_MP.b_over_a"] = (
        values["b"].div(nonzero_a).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    out["COND.substrate_MP.c_over_a"] = (
        values["c"].div(nonzero_a).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    volume_per_site = (
        values["volume"]
        .div(nonzero_sites)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out["COND.substrate_MP.volume_per_site_ang3"] = volume_per_site
    out["COND.substrate_MP.log1p_volume_per_site"] = np.log1p(
        volume_per_site.clip(lower=0.0)
    )

    crystal_system = df.get(
        "substrate_mp_crystal_system_inferred", pd.Series("", index=df.index)
    ).fillna("").astype(str).str.strip().str.lower()
    for system in [
        "cubic",
        "hexagonal",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "triclinic",
        "trigonal",
    ]:
        out[f"COND.substrate_MP.crystal_{system}"] = crystal_system.eq(system).astype(float)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    manifest = {
        "mode": "mp_lattice",
        "n_features": int(out.shape[1]),
        "available_rows": int(available.sum()),
        "phase_review_required_rows": int(
            out["COND.substrate_MP.phase_review_required"].sum()
        ),
        "feature_columns": out.columns.tolist(),
    }
    return out, manifest


def make_condition_features(
    df: pd.DataFrame,
    include_extra_categorical: bool,
    substrate_feature_mode: str,
    condition_categorical_mode: str,
    cache_dir: Path,
    material_magpie: pd.DataFrame | None = None,
    substrate_top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(index=df.index)
    if condition_categorical_mode != "substrate_only":
        pressure = pd.to_numeric(
            df.get("pressure_gpa_standardized", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        field = pd.to_numeric(
            df.get("magnetic_field_t_standardized", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        out["COND.pressure_GPa"] = pressure
        out["COND.magnetic_field_T"] = field
        out["COND.pressure_missing"] = pressure.isna().astype(float)
        out["COND.magnetic_field_missing"] = field.isna().astype(float)
        out["COND.is_high_pressure_gt1GPa"] = (pressure > 1.0).astype(float)

    manifest: dict[str, Any] = {"substrate_feature_mode": substrate_feature_mode}
    categorical: list[str] = []
    if condition_categorical_mode == "compact" and "criterion_raw" in df.columns:
        criterion_features, criterion_manifest = compact_criterion_features(df["criterion_raw"])
        out = out.join(criterion_features)
        manifest["criterion"] = criterion_manifest
    elif condition_categorical_mode == "raw":
        categorical.append("criterion_raw")
        manifest["criterion"] = {"mode": "raw"}
    elif condition_categorical_mode == "measurement":
        categorical.append("tc_measurement_method")
        manifest["criterion"] = {"mode": "standardized_measurement_method"}
    elif condition_categorical_mode == "substrate_only":
        manifest["criterion"] = {"mode": "excluded_substrate_only"}
        manifest["experimental_numeric"] = {"mode": "excluded_substrate_only"}
    else:
        manifest["criterion"] = {"mode": "excluded_numeric_condition_only"}
    if substrate_feature_mode == "raw_onehot":
        categorical.append("substrate")
    elif substrate_feature_mode in {
        "magpie",
        "status",
        "magpie_only",
        "film_status",
        "physical",
        "physical_only",
        "interface",
        "interface_top",
        "mp_lattice",
        "physical_mp",
    } and "substrate" in df.columns:
        base_substrate_mode = (
            "film_status"
            if substrate_feature_mode == "mp_lattice"
            else "physical"
            if substrate_feature_mode == "physical_mp"
            else substrate_feature_mode
        )
        substrate_features, substrate_manifest = make_substrate_features(
            df["substrate"],
            cache_dir=cache_dir,
            mode=base_substrate_mode,
            synthesis_values=df.get("synthesis_method", pd.Series("<empty>", index=df.index)),
            material_formulas=df.get("formula_sc", pd.Series("", index=df.index)),
            material_magpie=material_magpie,
            top_n=substrate_top_n,
        )
        if substrate_feature_mode in {"mp_lattice", "physical_mp"}:
            mp_features, mp_manifest = make_mp_substrate_lattice_features(df)
            substrate_features = substrate_features.join(mp_features)
            substrate_manifest["mode"] = substrate_feature_mode
            substrate_manifest["mp_lattice"] = mp_manifest
            substrate_manifest["n_features"] = int(substrate_features.shape[1])
        out = out.join(substrate_features)
        manifest["substrate"] = substrate_manifest
    elif substrate_feature_mode == "none":
        manifest["substrate"] = {"mode": "none", "n_features": 0}
    else:
        manifest["substrate"] = {"mode": substrate_feature_mode, "n_features": 0, "missing_column": True}

    if include_extra_categorical:
        categorical += ["tc_measurement_method", "synthesis_method"]
    for col in categorical:
        if col not in df.columns:
            continue
        values = clean_text_series(df[col])
        dummies = pd.get_dummies(values, prefix=f"COND.{col}", dtype=float)
        out = out.join(dummies)
        if col == "substrate":
            manifest["substrate"] = {
                "mode": "raw_onehot",
                "raw_unique_values": int(values.nunique()),
                "n_features": int(dummies.shape[1]),
            }
    manifest["n_condition_features"] = int(out.shape[1])
    return out, manifest


def prepare_edos_features(df: pd.DataFrame, edos_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not edos_path.exists():
        raise FileNotFoundError(f"eDOS feature table does not exist: {edos_path}")
    edos = pd.read_csv(edos_path)
    edos = edos.drop_duplicates("record_id", keep="first")
    merged = df[["record_id"]].merge(edos, on="record_id", how="left")

    bin_cols = sorted([col for col in merged.columns if col.startswith("edos_bin_")])
    numeric_edos_cols = [
        col
        for col in merged.columns
        if col.startswith("edos_")
        and not col.startswith("edos_cif_")
        and col not in {"edos_formula"}
    ]
    wanted: list[str] = []
    for col in [*SUMMARY_EDOS_COLUMNS, *FERMI_DERIVED_EDOS_COLUMNS, *numeric_edos_cols, *bin_cols]:
        if col in merged.columns and col not in wanted:
            wanted.append(col)
    out = merged[wanted].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    metadata_cols = [col for col in ["edos_cif_source", "edos_cif_relpath"] if col in merged.columns]
    if metadata_cols:
        out = pd.concat([merged[metadata_cols].fillna("missing"), out], axis=1)

    source_counts = {}
    if "edos_cif_source" in merged.columns:
        source_counts = merged["edos_cif_source"].fillna("missing").value_counts(dropna=False).to_dict()
    manifest = {
        "edos_features_path": str(edos_path.resolve()),
        "n_rows_in_edos_table": int(len(edos)),
        "n_rows_after_merge": int(len(merged)),
        "n_summary_features": int(len([c for c in SUMMARY_EDOS_COLUMNS if c in out.columns])),
        "n_fermi_derived_features": int(len([c for c in FERMI_DERIVED_EDOS_COLUMNS if c in out.columns])),
        "n_bin_features": int(len(bin_cols)),
        "edos_source_counts": source_counts,
        "edos_available_rows": int((out.get("edos_available", pd.Series(0.0, index=out.index)) > 0).sum()),
    }
    return out, manifest


def derive_phdos_physical_features(phdos: pd.DataFrame, bin_cols: list[str]) -> pd.DataFrame:
    """Build superconductivity-oriented descriptors from a 0-1000 cm^-1 phDOS curve."""
    derived = pd.DataFrame(index=phdos.index)
    if not bin_cols:
        return derived

    cm_to_k = 1.438776877
    eps = 1.0e-12
    values = phdos[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1)
    prob = np.divide(
        values,
        totals[:, None],
        out=np.zeros_like(values, dtype=float),
        where=totals[:, None] > 0,
    )

    if len(bin_cols) > 1:
        freq = np.linspace(0.0, 1000.0, len(bin_cols), dtype=float)
    else:
        freq = np.array([0.0], dtype=float)
    freq_pos_mask = freq > 0
    freq_pos = freq[freq_pos_mask]
    prob_pos = prob[:, freq_pos_mask]
    pos_weight = prob_pos.sum(axis=1)
    prob_pos_norm = np.divide(
        prob_pos,
        pos_weight[:, None],
        out=np.zeros_like(prob_pos, dtype=float),
        where=pos_weight[:, None] > 0,
    )

    log_freq = np.exp((prob_pos_norm * np.log(freq_pos + eps)).sum(axis=1))
    inv_freq = (prob_pos_norm / (freq_pos + eps)).sum(axis=1)
    inv_freq2 = (prob_pos_norm / ((freq_pos + eps) ** 2)).sum(axis=1)
    shifted = freq + 20.0
    softness_m1 = (prob / shifted).sum(axis=1)
    softness_m2 = (prob / (shifted**2)).sum(axis=1)

    centroid = pd.to_numeric(
        phdos.get("phdos_centroid_cm-1", pd.Series(0.0, index=phdos.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    rms = pd.to_numeric(
        phdos.get("phdos_rms_freq_cm-1", pd.Series(0.0, index=phdos.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    width = pd.to_numeric(
        phdos.get("phdos_width_cm-1", pd.Series(0.0, index=phdos.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    q90 = pd.to_numeric(
        phdos.get("phdos_q90_cm-1", pd.Series(0.0, index=phdos.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    support_max = pd.to_numeric(
        phdos.get("phdos_support_max_cm-1", pd.Series(0.0, index=phdos.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    derived["phdos_log_freq_cm-1"] = np.where(pos_weight > 0, log_freq, 0.0)
    derived["phdos_log_temp_k"] = derived["phdos_log_freq_cm-1"] * cm_to_k
    derived["phdos_ad_prefactor_k"] = derived["phdos_log_temp_k"] / 1.2
    derived["phdos_centroid_temp_k"] = centroid * cm_to_k
    derived["phdos_rms_temp_k"] = rms * cm_to_k
    derived["phdos_q90_temp_k"] = q90 * cm_to_k
    derived["phdos_support_max_temp_k"] = support_max * cm_to_k
    derived["phdos_mean_inv_freq_cm"] = np.where(pos_weight > 0, inv_freq, 0.0)
    derived["phdos_mean_inv_freq2_cm2"] = np.where(pos_weight > 0, inv_freq2, 0.0)
    derived["phdos_softness_m1_shift20"] = softness_m1
    derived["phdos_softness_m2_shift20"] = softness_m2
    derived["phdos_freq_variation_coeff"] = np.divide(
        width,
        np.abs(centroid) + eps,
        out=np.zeros_like(width, dtype=float),
        where=np.abs(centroid) > 0,
    )
    derived["phdos_centroid_sq_cm-2"] = centroid**2
    derived["phdos_rms_sq_cm-2"] = rms**2
    derived["phdos_effective_bin_count"] = np.divide(
        1.0,
        (prob**2).sum(axis=1),
        out=np.zeros(len(prob), dtype=float),
        where=(prob**2).sum(axis=1) > 0,
    )
    derived["phdos_effective_bin_fraction"] = derived["phdos_effective_bin_count"] / max(len(bin_cols), 1)
    derived["phdos_spectral_roughness"] = np.abs(np.diff(prob, axis=1)).sum(axis=1)

    def frac(lo: float, hi: float, include_hi: bool = False) -> np.ndarray:
        if include_hi:
            mask = (freq >= lo) & (freq <= hi)
        else:
            mask = (freq >= lo) & (freq < hi)
        return prob[:, mask].sum(axis=1)

    frac_0_50 = frac(0.0, 50.0)
    frac_50_100 = frac(50.0, 100.0)
    frac_100_200 = frac(100.0, 200.0)
    frac_200_400 = frac(200.0, 400.0)
    frac_400_700 = frac(400.0, 700.0)
    frac_700_1000 = frac(700.0, 1000.0, include_hi=True)
    frac_0_120 = frac(0.0, 120.0)
    frac_120_300 = frac(120.0, 300.0)
    frac_300_1000 = frac(300.0, 1000.0, include_hi=True)
    acoustic = frac(0.0, 200.0)
    optical = frac(200.0, 1000.0, include_hi=True)
    stiff = frac(400.0, 1000.0, include_hi=True)

    derived["phdos_frac_0_50"] = frac_0_50
    derived["phdos_frac_50_100"] = frac_50_100
    derived["phdos_frac_100_200"] = frac_100_200
    derived["phdos_frac_200_400"] = frac_200_400
    derived["phdos_frac_400_700"] = frac_400_700
    derived["phdos_frac_700_1000"] = frac_700_1000
    derived["phdos_frac_0_120"] = frac_0_120
    derived["phdos_frac_120_300"] = frac_120_300
    derived["phdos_frac_300_1000"] = frac_300_1000
    derived["phdos_acoustic_to_optical_ratio"] = acoustic / (optical + eps)
    derived["phdos_optical_to_acoustic_ratio"] = optical / (acoustic + eps)
    derived["phdos_soft_to_stiff_ratio"] = (frac_0_50 + frac_50_100) / (stiff + eps)
    derived["phdos_low_to_high_ratio"] = acoustic / (stiff + eps)
    derived["phdos_soft_mode_score"] = (frac_0_50 + frac_50_100) * softness_m1
    derived["phdos_hard_mode_score"] = stiff * np.maximum(q90, support_max)

    low_mask = (freq >= 20.0) & (freq <= 120.0)
    x_low = freq[low_mask]
    x2_low = x_low**2
    low_slope = np.zeros(len(prob), dtype=float)
    debye_coeff = np.zeros(len(prob), dtype=float)
    debye_r2 = np.zeros(len(prob), dtype=float)
    largest_gap = np.zeros(len(prob), dtype=float)
    largest_gap_frac = np.zeros(len(prob), dtype=float)
    peak_count = np.zeros(len(prob), dtype=float)
    for idx, row in enumerate(prob):
        y_low = row[low_mask]
        if y_low.sum() > 0 and len(x_low) > 1:
            low_slope[idx] = float(np.polyfit(x_low, y_low, 1)[0])
            denom = float(np.dot(x2_low, x2_low))
            if denom > 0:
                coeff = float(np.dot(x2_low, y_low) / denom)
                pred = coeff * x2_low
                ss_tot = float(((y_low - y_low.mean()) ** 2).sum())
                ss_res = float(((y_low - pred) ** 2).sum())
                debye_coeff[idx] = coeff
                debye_r2[idx] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        nonzero_freq = freq[row > 1.0e-8]
        if len(nonzero_freq) > 1:
            gaps = np.diff(nonzero_freq)
            largest_gap[idx] = float(gaps.max())
            span = float(nonzero_freq.max() - nonzero_freq.min())
            largest_gap_frac[idx] = largest_gap[idx] / (span + eps) if span > 0 else 0.0

        ymax = float(row.max())
        if ymax > 0:
            thresh = 0.10 * ymax
            count = 0
            for j in range(len(row)):
                left = row[j - 1] if j > 0 else -np.inf
                right = row[j + 1] if j < len(row) - 1 else -np.inf
                if row[j] >= thresh and row[j] >= left and row[j] >= right and (row[j] > left or row[j] > right):
                    count += 1
            peak_count[idx] = float(count)

    derived["phdos_largest_gap_cm-1"] = largest_gap
    derived["phdos_largest_gap_fraction"] = largest_gap_frac
    derived["phdos_peak_count_10pct"] = peak_count
    derived["phdos_lowfreq_slope_20_120"] = low_slope
    derived["phdos_lowfreq_debye_coeff_20_120"] = debye_coeff
    derived["phdos_lowfreq_debye_r2_20_120"] = debye_r2
    return derived.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_phdos_features(df: pd.DataFrame, phdos_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not phdos_path.exists():
        raise FileNotFoundError(f"phonon-DOS feature table does not exist: {phdos_path}")
    phdos = pd.read_csv(phdos_path)

    if "record_id" in phdos.columns:
        phdos = phdos.drop_duplicates("record_id", keep="first")
        merged = df[["record_id", "primary_cif_relpath"]].merge(phdos, on="record_id", how="left")
    elif "primary_cif_relpath" in phdos.columns:
        phdos = phdos.drop_duplicates("primary_cif_relpath", keep="first").copy()
        phdos["primary_cif_relpath"] = phdos["primary_cif_relpath"].astype(str).str.replace("\\", "/", regex=False)
        keys = df[["record_id", "primary_cif_relpath"]].copy()
        keys["primary_cif_relpath"] = keys["primary_cif_relpath"].astype(str).str.replace("\\", "/", regex=False)
        merged = keys.merge(phdos, on="primary_cif_relpath", how="left")
    else:
        raise RuntimeError("phDOS feature table must contain record_id or primary_cif_relpath.")

    bin_cols = sorted([col for col in merged.columns if col.startswith("phdos_bin_")])
    numeric_phdos_cols = [
        col
        for col in merged.columns
        if col.startswith("phdos_") and col not in {"phdos_error"}
    ]
    wanted: list[str] = []
    for col in [*SUMMARY_PHDOS_COLUMNS, *numeric_phdos_cols, *bin_cols]:
        if col in merged.columns and col not in wanted:
            wanted.append(col)

    out = merged[wanted].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    physical = derive_phdos_physical_features(out, bin_cols)
    if not physical.empty:
        out = pd.concat([out, physical], axis=1)
    if "phdos_error" in merged.columns:
        out = pd.concat([merged[["phdos_error"]].fillna(""), out], axis=1)

    manifest = {
        "phdos_features_path": str(phdos_path.resolve()),
        "n_rows_in_phdos_table": int(len(phdos)),
        "n_rows_after_merge": int(len(merged)),
        "n_summary_features": int(len([c for c in SUMMARY_PHDOS_COLUMNS if c in out.columns])),
        "n_physical_features": int(len([c for c in PHYSICAL_PHDOS_COLUMNS if c in out.columns])),
        "n_bin_features": int(len(bin_cols)),
        "phdos_available_rows": int((out.get("phdos_available", pd.Series(0.0, index=out.index)) > 0).sum()),
    }
    return out, manifest


def prepare_feature_frame(
    df: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    edos_path: Path,
    phdos_path: Path,
    soap_n_jobs: int,
    skip_dsoap: bool,
    include_dsoap: bool,
    include_extra_categorical: bool,
    substrate_feature_mode: str,
    condition_categorical_mode: str,
    substrate_top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy().reset_index(drop=True)
    y = finite_float_series(work["tc_k_standardized"])
    keep = y.notna()
    work = work.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    work["tc"] = y.astype(float)
    work["formula_sc"] = work.apply(clean_formula_for_magpie, axis=1)
    work["chemical_composition_sc"] = work["formula_sc"].map(chemical_system)
    work["weight"] = 1.0 / work.groupby("formula_sc")["formula_sc"].transform("size")

    cache_dir = output_dir / "feature_cache"
    unique_formulas = pd.Series(sorted(work["formula_sc"].dropna().astype(str).unique()))
    magpie_unique = calculate_magpie_features(unique_formulas, cache_dir / "magpie_by_formula.pkl")
    magpie = magpie_unique.set_index("formula_sc").loc[work["formula_sc"]].reset_index(drop=True)
    magpie = magpie.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    paths, missing_cif_rows = resolve_cif_paths(work, dataset_root)
    cif_override_manifest: dict[str, Any] = {"enabled": False}
    if "cif_override_path" in work.columns:
        override_values = work["cif_override_path"].fillna("").astype(str).str.strip()
        override_used: list[bool] = []
        override_missing = 0
        resolved_paths: list[Path] = []
        for original_path, override_value in zip(paths, override_values):
            override_path = Path(override_value) if override_value else Path("")
            use_override = bool(override_value) and override_path.exists()
            if use_override:
                resolved_paths.append(override_path)
            else:
                resolved_paths.append(original_path)
                if override_value:
                    override_missing += 1
            override_used.append(use_override)
        paths = resolved_paths
        work["cif_override_used"] = np.asarray(override_used, dtype=bool)
        missing_cif_rows = int(sum(not str(path) for path in paths))
        cif_override_manifest = {
            "enabled": True,
            "requested_rows": int(override_values.ne("").sum()),
            "used_rows": int(sum(override_used)),
            "nonexistent_override_rows": int(override_missing),
            "fallback_to_primary_rows": int(len(paths) - sum(override_used)),
        }
    work["resolved_cif_path"] = [str(path) for path in paths]

    soap_names: list[str] = []
    structure_aux = pd.DataFrame(index=work.index)
    soap_df = pd.DataFrame(index=work.index)
    structure_meta: dict[str, Any] = {"requested": bool(include_dsoap), "skipped": True}
    if include_dsoap:
        soap_matrix, soap_names, structure_aux, structure_meta = calculate_structure_features(
            paths=paths,
            formulas=work["formula_sc"],
            cache_npz=cache_dir / f"structure_features_skip{int(skip_dsoap)}.npz",
            cache_meta=cache_dir / f"structure_features_skip{int(skip_dsoap)}.json",
            soap_n_jobs=soap_n_jobs,
            skip_dsoap=skip_dsoap,
        )
        soap_df = pd.DataFrame(soap_matrix, columns=soap_names).fillna(0.0)
        structure_aux = structure_aux.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    condition, condition_manifest = make_condition_features(
        work,
        include_extra_categorical=include_extra_categorical,
        substrate_feature_mode=substrate_feature_mode,
        condition_categorical_mode=condition_categorical_mode,
        cache_dir=cache_dir,
        material_magpie=magpie,
        substrate_top_n=substrate_top_n,
    )
    edos, edos_manifest = prepare_edos_features(work, edos_path)
    phdos, phdos_manifest = prepare_phdos_features(work, phdos_path)

    feature_frame = pd.concat(
        [
            work,
            magpie.add_prefix(""),
            structure_aux,
            soap_df,
            condition,
            edos,
            phdos,
        ],
        axis=1,
    )

    manifest = {
        "rows": int(len(feature_frame)),
        "rows_by_fixed_split": feature_frame["fixed_split"].value_counts(dropna=False).to_dict(),
        "unique_formula_sc": int(feature_frame["formula_sc"].nunique()),
        "unique_chemical_composition_sc": int(feature_frame["chemical_composition_sc"].nunique()),
        "missing_cif_rows": int(missing_cif_rows),
        "n_magpie_features": int(len([c for c in feature_frame.columns if c.startswith("MAGPIE.")])),
        "n_condition_features": int(len([c for c in feature_frame.columns if c.startswith("COND.")])),
        "n_substrate_condition_features": int(
            len([c for c in feature_frame.columns if c.startswith("COND.substrate")])
        ),
        "n_soap_features": int(len(soap_names)),
        "n_structure_aux_features": int(len([c for c in STRUCTURE_AUX_FEATURES if c in feature_frame.columns])),
        "condition": condition_manifest,
        "edos": edos_manifest,
        "phdos": phdos_manifest,
        "structure_meta": structure_meta,
        "cif_path_override": cif_override_manifest,
    }
    return feature_frame, manifest


def make_group_cv_splits(
    df: pd.DataFrame,
    n_reps: int,
    train_frac: float,
    random_seed: int,
) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(train_size=train_frac, n_splits=n_reps, random_state=random_seed)
    out = pd.DataFrame(index=df.index)
    groups = df["chemical_composition_sc"].to_numpy()
    x_dummy = np.zeros((len(df), 1))
    for rep, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, df["tc"], groups=groups)):
        values = np.full(len(df), "", dtype=object)
        values[train_idx] = "train"
        values[test_idx] = "test"
        out[f"CV_{rep}"] = values
    return out


def fixed_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["CV_0"] = df["fixed_split"].astype(str).to_numpy()
    return out


def read_reusable_cv_splits(df: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    if not split_path.exists():
        raise FileNotFoundError(f"CV split file does not exist: {split_path}")
    split_table = pd.read_csv(split_path)
    if "record_id" not in split_table.columns:
        raise RuntimeError("CV split file must contain record_id.")
    if split_table["record_id"].duplicated().any():
        raise RuntimeError("CV split file contains duplicate record_id values.")
    cv_cols = sorted(
        [col for col in split_table.columns if re.fullmatch(r"CV_\d+", col)],
        key=lambda col: int(col.split("_")[1]),
    )
    if not cv_cols:
        raise RuntimeError("CV split file contains no CV_<n> columns.")
    indexed = split_table.set_index("record_id")
    missing = sorted(set(df["record_id"]) - set(indexed.index))
    if missing:
        raise RuntimeError(
            f"CV split file is missing {len(missing)} retained record ids; first: {missing[:5]}"
        )
    out = indexed.loc[df["record_id"], cv_cols].reset_index(drop=True)
    invalid: dict[str, list[str]] = {}
    for col in cv_cols:
        values = set(out[col].fillna("").astype(str).unique())
        bad = sorted(values - {"train", "test"})
        if bad:
            invalid[col] = bad
    if invalid:
        raise RuntimeError(f"CV split file has invalid labels: {invalid}")
    return out


def feature_tokens(feature_set: str) -> set[str]:
    return {item.strip().upper() for item in feature_set.split("+") if item.strip()}


def edos_pca_components(tokens: set[str]) -> list[int]:
    comps: list[int] = []
    for token in tokens:
        if not token.startswith("EDOS_PCA"):
            continue
        suffix = token.replace("EDOS_PCA", "", 1)
        if not suffix.isdigit():
            raise ValueError(f"Use EDOS_PCA<N>, for example EDOS_PCA20, not {token}")
        comps.append(int(suffix))
    return sorted(set(comps))


def phdos_pca_components(tokens: set[str]) -> list[int]:
    comps: list[int] = []
    for token in tokens:
        if not token.startswith("PHDOS_PCA"):
            continue
        suffix = token.replace("PHDOS_PCA", "", 1)
        if not suffix.isdigit():
            raise ValueError(f"Use PHDOS_PCA<N>, for example PHDOS_PCA2, not {token}")
        comps.append(int(suffix))
    return sorted(set(comps))


def dsoap_pca_components(tokens: set[str]) -> list[int]:
    comps: list[int] = []
    for token in tokens:
        if not token.startswith("DSOAP_PCA"):
            continue
        suffix = token.replace("DSOAP_PCA", "", 1)
        if not suffix.isdigit():
            raise ValueError(f"Use DSOAP_PCA<N>, for example DSOAP_PCA128, not {token}")
        comps.append(int(suffix))
    return sorted(set(comps))


def selected_feature_columns(
    df: pd.DataFrame, feature_set: str, include_condition_features: bool
) -> list[str]:
    tokens = feature_tokens(feature_set)
    dsoap_pca = dsoap_pca_components(tokens)
    cols: list[str] = []
    if "MAGPIE" in tokens:
        cols += [c for c in df.columns if c.startswith("MAGPIE.")]
    if "DSOAP" in tokens:
        cols += [c for c in df.columns if c.startswith("SOAP_")]
        cols += [c for c in STRUCTURE_AUX_FEATURES if c in df.columns]
    if dsoap_pca:
        cols += [c for c in STRUCTURE_AUX_FEATURES if c in df.columns]
    if include_condition_features:
        cols += [c for c in df.columns if c.startswith("COND.")]

    bin_cols = sorted([c for c in df.columns if c.startswith("edos_bin_")])
    phdos_bin_cols = sorted([c for c in df.columns if c.startswith("phdos_bin_")])
    summary_cols = [c for c in SUMMARY_EDOS_COLUMNS if c in df.columns]
    phdos_summary_cols = [c for c in SUMMARY_PHDOS_COLUMNS if c in df.columns]
    phdos_physical_cols = [c for c in PHYSICAL_PHDOS_COLUMNS if c in df.columns]
    if "EDOS_SUMMARY_BINS" in tokens:
        cols += summary_cols + bin_cols
    else:
        if "EDOS_SUMMARY" in tokens:
            cols += summary_cols
        if "EDOS_NEAR_FERMI" in tokens:
            cols += [c for c in NEAR_FERMI_EDOS_COLUMNS if c in df.columns]
        if "EDOS_BANDS" in tokens:
            cols += [c for c in BAND_EDOS_COLUMNS if c in df.columns]
        if "EDOS_FERMI_DERIVED" in tokens:
            cols += [c for c in FERMI_DERIVED_EDOS_COLUMNS if c in df.columns]
        if "EDOS_BINS" in tokens:
            available = ["edos_available"] if "edos_available" in df.columns else []
            cols += available + bin_cols
        if edos_pca_components(tokens) and "edos_available" in df.columns:
            cols += ["edos_available"]

    if "PHDOS_SUMMARY_BINS" in tokens:
        cols += phdos_summary_cols + phdos_bin_cols
    else:
        if "PHDOS_SUMMARY" in tokens:
            cols += phdos_summary_cols
        if "PHDOS_PHYSICAL" in tokens:
            cols += phdos_summary_cols + phdos_physical_cols
        if "PHDOS_BINS" in tokens:
            available = ["phdos_available"] if "phdos_available" in df.columns else []
            cols += available + phdos_bin_cols
        if phdos_pca_components(tokens) and "phdos_available" in df.columns:
            cols += ["phdos_available"]

    deduped: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col not in seen:
            seen.add(col)
            deduped.append(col)
    if not deduped:
        raise RuntimeError(f"No features selected for {feature_set}")
    if "DSOAP" in tokens and not any(c.startswith("SOAP_") for c in deduped):
        raise RuntimeError(f"{feature_set} requested DSOAP but no SOAP_ features are available")
    return deduped


def append_split_pca(
    x_train: np.ndarray,
    x_test: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    component_counts: list[int],
    random_seed: int,
    name_prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not component_counts:
        return x_train, x_test, []
    if source_train.shape[1] == 0:
        raise RuntimeError(f"{name_prefix.upper()}_PCA requested but no source columns are available.")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(source_train)
    test_scaled = scaler.transform(source_test)

    pca_train_blocks: list[np.ndarray] = []
    pca_test_blocks: list[np.ndarray] = []
    names: list[str] = []
    max_components = min(train_scaled.shape[0], train_scaled.shape[1])
    for requested in component_counts:
        n_components = min(int(requested), max_components)
        pca = PCA(n_components=n_components, random_state=random_seed)
        pca_train_blocks.append(pca.fit_transform(train_scaled))
        pca_test_blocks.append(pca.transform(test_scaled))
        names.extend([f"{name_prefix}_pca{requested}_{idx:02d}" for idx in range(n_components)])

    return (
        np.hstack([x_train, *pca_train_blocks]).astype(np.float32, copy=False),
        np.hstack([x_test, *pca_test_blocks]).astype(np.float32, copy=False),
        names,
    )


def unweighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    err = y_pred - y_true
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else float("nan")
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    log_err = log_pred - log_true
    log_denom = float(np.sum((log_true - log_true.mean()) ** 2))
    log_r2 = (
        1.0 - float(np.sum(log_err**2)) / log_denom
        if log_denom > 0
        else float("nan")
    )
    return {
        "MAE_unweighted": float(np.mean(np.abs(err))),
        "RMSE_unweighted": float(math.sqrt(np.mean(err**2))),
        "r2_unweighted": r2,
        "MSLE_unweighted": float(np.mean(log_err**2)),
        "r2_log1p_unweighted": log_r2,
    }


def restricted_sinh_with_upper_bound(
    x: np.ndarray, upper_bound_k: float
) -> np.ndarray:
    norm = np.arcsinh(1 / 2) * 10
    transformed = np.asarray(x, dtype=float) * norm
    upper = max(float(upper_bound_k), 0.0)
    transformed = np.clip(
        transformed,
        np.arcsinh(0),
        np.arcsinh(upper / 2),
    )
    return np.sinh(transformed) * 2


def fit_predict_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_test: np.ndarray,
    random_seed: int,
    xgb_n_jobs: int,
    prediction_upper_bound_k: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from xgboost import XGBRegressor

    model = XGBRegressor(n_jobs=xgb_n_jobs, random_state=random_seed)
    regressor = Pipeline([("StandardScaler", StandardScaler()), ("model", model)])
    target_transformer = FunctionTransformer(
        func=restricted_arcsinh,
        inverse_func=restricted_sinh_with_upper_bound,
        inv_kw_args={"upper_bound_k": prediction_upper_bound_k},
        check_inverse=False,
    )
    regr = TransformedTargetRegressor(regressor=regressor, transformer=target_transformer)
    regr.fit(x_train, y_train, model__sample_weight=w_train)
    pred_train = np.maximum(regr.predict(x_train), 0.0)
    pred_test = np.maximum(regr.predict(x_test), 0.0)
    return pred_train, pred_test, regr


def select_topk_features_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    topk: int,
    random_seed: int,
    xgb_n_jobs: int,
) -> np.ndarray:
    n_features = int(x_train.shape[1])
    if topk <= 0 or topk >= n_features:
        return np.arange(n_features, dtype=int)

    from xgboost import XGBRegressor

    selector = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=xgb_n_jobs,
        random_state=random_seed,
    )
    selector.fit(x_train, restricted_arcsinh(y_train), sample_weight=w_train)
    importance = np.asarray(getattr(selector, "feature_importances_", None), dtype=float)
    if importance.shape[0] != n_features or not np.isfinite(importance).any() or float(importance.sum()) <= 0:
        variance = np.nan_to_num(np.var(x_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        ranking_score = variance
    else:
        ranking_score = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.argsort(ranking_score)[::-1][:topk]
    return np.sort(selected.astype(int))


def train_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    cv_cols: list[str],
    output_dir: Path,
    random_seed: int,
    xgb_n_jobs: int,
    include_condition_features: bool,
    feature_select_topk: int,
    prediction_upper_bound_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tokens = feature_tokens(feature_set)
    pca_components = edos_pca_components(tokens)
    phdos_pca = phdos_pca_components(tokens)
    dsoap_pca = dsoap_pca_components(tokens)
    feature_cols = selected_feature_columns(
        df, feature_set, include_condition_features=include_condition_features
    )
    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    bin_cols = sorted([c for c in df.columns if c.startswith("edos_bin_")])
    bin_all = (
        df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if pca_components
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    phdos_bin_cols = sorted([c for c in df.columns if c.startswith("phdos_bin_")])
    phdos_bin_all = (
        df[phdos_bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if phdos_pca
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    soap_cols = sorted([c for c in df.columns if c.startswith("SOAP_")])
    soap_all = (
        df[soap_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if dsoap_pca
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    y_all = df["tc"].to_numpy(dtype=float)
    w_all = df["weight"].to_numpy(dtype=float)

    prediction_cols = [
        "record_id",
        "formula_standardized",
        "formula_reduced",
        "formula_sc",
        "chemical_composition_sc",
        "tc",
        "fixed_split",
        "primary_cif_relpath",
        "resolved_cif_path",
        "cif_override_used",
        "cif_mapping_selection",
        "cif_mapping_cif_match_type",
        "cif_mapping_source_path",
        "weight",
        "edos_available",
        "edos_cif_source",
        "edos_cif_relpath",
        "phdos_available",
    ]
    predictions_base = df[[c for c in prediction_cols if c in df.columns]].copy()

    score_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    pca_feature_cols: list[str] = []
    phdos_pca_feature_cols: list[str] = []
    dsoap_pca_feature_cols: list[str] = []
    for rep, cv_col in enumerate(cv_cols):
        train_mask = df[cv_col].eq("train").to_numpy()
        test_mask = df[cv_col].eq("test").to_numpy()
        train_groups = set(df.loc[train_mask, "chemical_composition_sc"])
        test_groups = set(df.loc[test_mask, "chemical_composition_sc"])
        overlap = sorted(train_groups & test_groups)

        x_train, x_test = x_all[train_mask], x_all[test_mask]
        if pca_components:
            x_train, x_test, pca_names = append_split_pca(
                x_train=x_train,
                x_test=x_test,
                source_train=bin_all[train_mask],
                source_test=bin_all[test_mask],
                component_counts=pca_components,
                random_seed=random_seed + rep,
                name_prefix="edos",
            )
            if not pca_feature_cols:
                pca_feature_cols = pca_names
        if phdos_pca:
            x_train, x_test, pca_names = append_split_pca(
                x_train=x_train,
                x_test=x_test,
                source_train=phdos_bin_all[train_mask],
                source_test=phdos_bin_all[test_mask],
                component_counts=phdos_pca,
                random_seed=random_seed + rep,
                name_prefix="phdos",
            )
            if not phdos_pca_feature_cols:
                phdos_pca_feature_cols = pca_names
        if dsoap_pca:
            x_train, x_test, pca_names = append_split_pca(
                x_train=x_train,
                x_test=x_test,
                source_train=soap_all[train_mask],
                source_test=soap_all[test_mask],
                component_counts=dsoap_pca,
                random_seed=random_seed + rep,
                name_prefix="dsoap",
            )
            if not dsoap_pca_feature_cols:
                dsoap_pca_feature_cols = pca_names
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        w_train, w_test = w_all[train_mask], w_all[test_mask]
        selected_idx = select_topk_features_xgb(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            topk=feature_select_topk,
            random_seed=random_seed + rep,
            xgb_n_jobs=xgb_n_jobs,
        )
        x_train = x_train[:, selected_idx]
        x_test = x_test[:, selected_idx]
        pred_train, pred_test, _ = fit_predict_xgb(
            x_train,
            y_train,
            w_train,
            x_test,
            random_seed=random_seed + rep,
            xgb_n_jobs=xgb_n_jobs,
            prediction_upper_bound_k=prediction_upper_bound_k,
        )

        for split_name, y_true, y_pred, weights, mask in [
            ("train", y_train, pred_train, w_train, train_mask),
            ("test", y_test, pred_test, w_test, test_mask),
        ]:
            weighted = metrics_3dsc(y_true, y_pred, weights)
            plain = unweighted_metrics(y_true, y_pred)
            row = {
                "features": feature_set,
                "model": "XGB",
                "repetition": rep,
                "split": split_name,
                "n_rows": int(mask.sum()),
                "n_features_used": int(len(selected_idx)),
                "n_chemical_systems": int(df.loc[mask, "chemical_composition_sc"].nunique()),
                "chemical_system_overlap": int(len(overlap)),
            }
            row.update(weighted)
            row.update(plain)
            score_rows.append(row)

        test_pred = predictions_base.loc[test_mask].copy()
        test_pred["features"] = feature_set
        test_pred["repetition"] = rep
        test_pred["tc_pred"] = pred_test
        test_pred["abs_error"] = np.abs(pred_test - y_test)
        pred_rows.append(test_pred)
        test_weighted = score_rows[-1]
        print(
            f"  {feature_set} rep={rep}: test weighted R2={test_weighted['r2']:.4f}, "
            f"unweighted R2={test_weighted['r2_unweighted']:.4f}, "
            f"MAE={test_weighted['MAE']:.3f}",
            flush=True,
        )

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    safe_name = feature_set.replace("+", "_")
    write_csv_xlsx(scores, output_dir / f"{safe_name}_scores_by_split.csv")
    write_csv_xlsx(predictions, output_dir / f"{safe_name}_test_predictions.csv")
    manifest_feature_cols = feature_cols + pca_feature_cols + phdos_pca_feature_cols + dsoap_pca_feature_cols
    manifest = {
        "feature_set": feature_set,
        "n_features": int(len(manifest_feature_cols)),
        "n_magpie_features": int(len([c for c in manifest_feature_cols if c.startswith("MAGPIE.")])),
        "n_condition_features": int(len([c for c in manifest_feature_cols if c.startswith("COND.")])),
        "n_soap_features": int(len([c for c in manifest_feature_cols if c.startswith("SOAP_")])),
        "n_dsoap_pca_features": int(len(dsoap_pca_feature_cols)),
        "dsoap_pca_components": dsoap_pca,
        "n_structure_aux_features": int(
            len([c for c in manifest_feature_cols if c in STRUCTURE_AUX_FEATURES])
        ),
        "n_edos_summary_features": int(
            len([c for c in manifest_feature_cols if c in SUMMARY_EDOS_COLUMNS])
        ),
        "n_edos_near_fermi_features": int(
            len([c for c in manifest_feature_cols if c in NEAR_FERMI_EDOS_COLUMNS])
        ),
        "n_edos_band_features": int(
            len([c for c in manifest_feature_cols if c in BAND_EDOS_COLUMNS])
        ),
        "n_edos_fermi_derived_features": int(
            len([c for c in manifest_feature_cols if c in FERMI_DERIVED_EDOS_COLUMNS])
        ),
        "n_edos_bin_features": int(len([c for c in manifest_feature_cols if c.startswith("edos_bin_")])),
        "n_edos_pca_features": int(len(pca_feature_cols)),
        "edos_pca_components": pca_components,
        "n_phdos_summary_features": int(
            len([c for c in manifest_feature_cols if c in SUMMARY_PHDOS_COLUMNS])
        ),
        "n_phdos_physical_features": int(
            len([c for c in manifest_feature_cols if c in PHYSICAL_PHDOS_COLUMNS])
        ),
        "n_phdos_bin_features": int(len([c for c in manifest_feature_cols if c.startswith("phdos_bin_")])),
        "n_phdos_pca_features": int(len(phdos_pca_feature_cols)),
        "phdos_pca_components": phdos_pca,
        "feature_columns": manifest_feature_cols,
    }
    return scores, predictions, manifest


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "MSLE",
        "MAE",
        "RMSE",
        "median_AE",
        "r2",
        "MAE_unweighted",
        "RMSE_unweighted",
        "r2_unweighted",
        "MSLE_unweighted",
        "r2_log1p_unweighted",
        "n_rows",
        "n_features_used",
        "n_chemical_systems",
        "chemical_system_overlap",
    ]
    grouped = scores.groupby(["dataset", "features", "model", "split"], dropna=False)
    agg = grouped[value_cols].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = [
        "_".join(part for part in col if part) if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    reps = grouped["repetition"].nunique().reset_index(name="n_reps")
    return agg.merge(reps, on=["dataset", "features", "model", "split"], how="left")


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    dataset_root = base_dir / args.dataset
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    include_dsoap = any(
        any(token == "DSOAP" or token.startswith("DSOAP_PCA") for token in feature_tokens(item))
        for item in feature_sets
    )

    data_csv = args.data_csv.resolve() if args.data_csv is not None else None
    raw_df = read_input_table(dataset_root, data_csv)
    mp_mapping_manifest: dict[str, Any] = {"enabled": False}
    if (
        args.substrate_feature_mode in {"mp_lattice", "physical_mp"}
        or args.exclude_unmatched_known_substrates
    ):
        raw_df, mp_mapping_manifest = attach_mp_substrate_mapping(
            raw_df,
            mapping_path=args.substrate_mp_mapping.resolve(),
            exclude_unmatched_known=args.exclude_unmatched_known_substrates,
        )
    cif_mapping_manifest: dict[str, Any] = {"enabled": False}
    if args.cif_path_mapping is not None:
        raw_df, cif_mapping_manifest = attach_cif_path_mapping(
            raw_df,
            mapping_path=args.cif_path_mapping.resolve(),
            path_column=args.cif_path_column,
        )
    high_tc_filter_manifest: dict[str, Any] = {"enabled": False}
    if args.require_positive_pressure_above_tc_k > 0:
        threshold = float(args.require_positive_pressure_above_tc_k)
        tc = pd.to_numeric(raw_df["tc_k_standardized"], errors="coerce")
        pressure = pd.to_numeric(
            raw_df.get(
                "pressure_gpa_standardized",
                pd.Series(np.nan, index=raw_df.index),
            ),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        high_tc = tc > threshold
        remove = high_tc & ~pressure.gt(0.0)
        removed = raw_df.loc[
            remove,
            [
                col
                for col in [
                    "record_id",
                    "formula_standardized",
                    "tc_k_standardized",
                    "pressure_gpa_standardized",
                    "pressure_raw",
                ]
                if col in raw_df.columns
            ],
        ].copy()
        write_csv_xlsx(removed, output_dir / "excluded_unverified_high_tc_records.csv")
        before_rows = int(len(raw_df))
        raw_df = raw_df.loc[~remove].reset_index(drop=True)
        high_tc_filter_manifest = {
            "enabled": True,
            "threshold_k": threshold,
            "policy": "Tc above threshold requires finite pressure_gpa_standardized > 0",
            "rows_before": before_rows,
            "rows_after": int(len(raw_df)),
            "high_tc_rows_before": int(high_tc.sum()),
            "verified_positive_pressure_high_tc_rows_retained": int(
                (high_tc & pressure.gt(0.0)).sum()
            ),
            "unverified_or_nonpositive_pressure_high_tc_rows_removed": int(remove.sum()),
        }
    df, data_manifest = prepare_feature_frame(
        df=raw_df,
        dataset_root=dataset_root,
        output_dir=output_dir,
        edos_path=args.edos_features.resolve(),
        phdos_path=args.phdos_features.resolve(),
        soap_n_jobs=args.soap_n_jobs,
        skip_dsoap=args.skip_dsoap,
        include_dsoap=include_dsoap,
        include_extra_categorical=args.include_extra_categorical,
        substrate_feature_mode=args.substrate_feature_mode,
        condition_categorical_mode=args.condition_categorical_mode,
        substrate_top_n=args.substrate_top_n,
    )
    data_manifest["substrate_mp_mapping"] = mp_mapping_manifest
    data_manifest["cif_path_mapping"] = cif_mapping_manifest
    data_manifest["high_tc_pressure_filter"] = high_tc_filter_manifest
    if args.require_edos:
        before_rows = int(len(df))
        before_by_split = df["fixed_split"].value_counts(dropna=False).to_dict()
        edos_available = pd.to_numeric(
            df.get("edos_available", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        df = df.loc[edos_available > 0].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No rows remain after --require-edos filtering.")
        data_manifest["require_edos_filter"] = {
            "enabled": True,
            "before_rows": before_rows,
            "after_rows": int(len(df)),
            "removed_rows": int(before_rows - len(df)),
            "before_rows_by_fixed_split": before_by_split,
            "after_rows_by_fixed_split": df["fixed_split"].value_counts(dropna=False).to_dict(),
            "unique_formula_sc_after": int(df["formula_sc"].nunique()),
            "unique_chemical_composition_sc_after": int(df["chemical_composition_sc"].nunique()),
        }
    else:
        data_manifest["require_edos_filter"] = {"enabled": False}
    if args.require_phdos:
        before_rows = int(len(df))
        before_by_split = df["fixed_split"].value_counts(dropna=False).to_dict()
        phdos_available = pd.to_numeric(
            df.get("phdos_available", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        df = df.loc[phdos_available > 0].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No rows remain after --require-phdos filtering.")
        data_manifest["require_phdos_filter"] = {
            "enabled": True,
            "before_rows": before_rows,
            "after_rows": int(len(df)),
            "removed_rows": int(before_rows - len(df)),
            "before_rows_by_fixed_split": before_by_split,
            "after_rows_by_fixed_split": df["fixed_split"].value_counts(dropna=False).to_dict(),
            "unique_formula_sc_after": int(df["formula_sc"].nunique()),
            "unique_chemical_composition_sc_after": int(df["chemical_composition_sc"].nunique()),
        }
    else:
        data_manifest["require_phdos_filter"] = {"enabled": False}
    if args.require_zero_pressure_field:
        before_rows = int(len(df))
        before_by_split = df["fixed_split"].value_counts(dropna=False).to_dict()
        pressure = pd.to_numeric(
            df.get("pressure_gpa_standardized", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        field = pd.to_numeric(
            df.get("magnetic_field_t_standardized", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        df = df.loc[pressure.eq(0.0) & field.eq(0.0)].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No rows remain after --require-zero-pressure-field filtering.")
        data_manifest["require_zero_pressure_field_filter"] = {
            "enabled": True,
            "strict_equal_zero": True,
            "before_rows": before_rows,
            "after_rows": int(len(df)),
            "removed_rows": int(before_rows - len(df)),
            "pressure_standardized_missing_rows": int(pressure.isna().sum()),
            "magnetic_field_standardized_missing_rows": int(field.isna().sum()),
            "before_rows_by_fixed_split": before_by_split,
            "after_rows_by_fixed_split": df["fixed_split"].value_counts(dropna=False).to_dict(),
            "unique_formula_sc_after": int(df["formula_sc"].nunique()),
            "unique_chemical_composition_sc_after": int(df["chemical_composition_sc"].nunique()),
        }
    else:
        data_manifest["require_zero_pressure_field_filter"] = {"enabled": False}

    if args.cv_split_file is not None:
        splits = read_reusable_cv_splits(df, args.cv_split_file.resolve())
    elif args.split_mode == "fixed":
        splits = fixed_split_frame(df)
    else:
        splits = make_group_cv_splits(df, args.n_reps, args.train_frac, args.random_seed)
    df = pd.concat([df, splits], axis=1)
    cv_cols = [c for c in df.columns if c.startswith("CV_")]

    split_audit_rows: list[dict[str, Any]] = []
    for col in cv_cols:
        train = df[col].eq("train")
        test = df[col].eq("test")
        split_audit_rows.append(
            {
                "cv": col,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "train_chemical_systems": int(df.loc[train, "chemical_composition_sc"].nunique()),
                "test_chemical_systems": int(df.loc[test, "chemical_composition_sc"].nunique()),
                "chemical_system_overlap": int(
                    len(
                        set(df.loc[train, "chemical_composition_sc"])
                        & set(df.loc[test, "chemical_composition_sc"])
                    )
                ),
            }
        )
    split_audit = pd.DataFrame(split_audit_rows)
    write_csv_xlsx(split_audit, output_dir / "split_audit.csv")

    all_scores: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "base_dir": str(base_dir),
        "dataset_root": str(dataset_root),
        "data_csv": str(data_csv) if data_csv is not None else None,
        "output_dir": str(output_dir),
        "python_executable": sys.executable,
        "split_mode": args.split_mode,
        "n_reps": int(len(cv_cols)),
        "train_frac": float(args.train_frac),
        "random_seed": int(args.random_seed),
        "cv_split_file": (
            str(args.cv_split_file.resolve()) if args.cv_split_file is not None else None
        ),
        "cif_path_mapping": (
            str(args.cif_path_mapping.resolve())
            if args.cif_path_mapping is not None
            else None
        ),
        "cif_path_column": args.cif_path_column,
        "xgb_n_jobs": int(args.xgb_n_jobs),
        "prediction_upper_bound_k": float(args.prediction_upper_bound_k),
        "feature_select_topk": int(args.feature_select_topk),
        "feature_sets_requested": feature_sets,
        "include_condition_features": not args.exclude_condition_features,
        "substrate_feature_mode": args.substrate_feature_mode,
        "substrate_top_n": int(args.substrate_top_n),
        "substrate_mp_mapping": str(args.substrate_mp_mapping.resolve()),
        "exclude_unmatched_known_substrates": bool(
            args.exclude_unmatched_known_substrates
        ),
        "condition_categorical_mode": args.condition_categorical_mode,
        "protocol": (
            "Current-package adapter for 3DSC-style XGB: MAGPIE/optional DSOAP, "
            f"{'with' if not args.exclude_condition_features else 'without'} condition features, "
            "sample weights by formula_sc, restricted arcsinh target."
        ),
        "data": data_manifest,
        "feature_sets": {},
    }
    for feature_set in feature_sets:
        print(f"\n=== Training {feature_set} ===", flush=True)
        scores, predictions, feature_manifest = train_feature_set(
            df=df,
            feature_set=feature_set,
            cv_cols=cv_cols,
            output_dir=output_dir,
            random_seed=args.random_seed,
            xgb_n_jobs=args.xgb_n_jobs,
            include_condition_features=not args.exclude_condition_features,
            feature_select_topk=args.feature_select_topk,
            prediction_upper_bound_k=args.prediction_upper_bound_k,
        )
        scores.insert(0, "dataset", args.dataset)
        predictions.insert(0, "dataset", args.dataset)
        all_scores.append(scores)
        all_predictions.append(predictions)
        manifest["feature_sets"][feature_set] = feature_manifest

    scores_df = pd.concat(all_scores, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregate = aggregate_scores(scores_df)
    write_csv_xlsx(scores_df, output_dir / "summary_scores_by_split.csv")
    write_csv_xlsx(aggregate, output_dir / "summary_scores_aggregate.csv")
    write_csv_xlsx(predictions_df, output_dir / "test_predictions_all_feature_sets.csv")
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    view = aggregate[aggregate["split"] == "test"][
        [
            "dataset",
            "features",
            "n_reps",
            "MAE_mean",
            "RMSE_mean",
            "r2_mean",
            "MAE_unweighted_mean",
            "RMSE_unweighted_mean",
            "r2_unweighted_mean",
            "chemical_system_overlap_mean",
        ]
    ].copy()
    print("\nAggregate test scores:")
    print(view.to_string(index=False))
    print(f"\noutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
