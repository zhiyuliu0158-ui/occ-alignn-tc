"""Shared Tc regime, binning, and subset helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

TC_BIN_EDGES = [-0.1, 1, 5, 10, 20, 40, 80, 120, float("inf")]
TC_BIN_LABELS = ["<=1", "1-5", "5-10", "10-20", "20-40", "40-80", "80-120", ">120"]
PRESSURE_BIN_EDGES = [-0.1, 0, 1, 10, 50, 100, float("inf")]
PRESSURE_BIN_LABELS = ["0", "0-1", "1-10", "10-50", "50-100", ">100"]
FIELD_BIN_EDGES = [-0.1, 0, 0.01, 1, 10, float("inf")]
FIELD_BIN_LABELS = ["0", "0-0.01", "0.01-1", "1-10", ">10"]

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series aligned to frame.index."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def text_column(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    """Return a string series aligned to frame.index."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[column].fillna(default).astype(str)


def formula_element_amounts(formula: object) -> dict[str, float]:
    """Parse a simple chemical formula into approximate element amounts."""
    text = str(formula or "").replace(" ", "")
    amounts: dict[str, float] = {}
    for element, amount_text in _TOKEN_RE.findall(text):
        amount = float(amount_text) if amount_text else 1.0
        amounts[element] = amounts.get(element, 0.0) + amount
    return amounts


def is_hydride_formula(formula: object, min_h_fraction: float = 0.5) -> bool:
    """Return true for H-rich formulas such as H3S, PH3, or SbH4."""
    amounts = formula_element_amounts(formula)
    total = sum(amounts.values())
    if total <= 0 or "H" not in amounts:
        return False
    return amounts["H"] / total >= min_h_fraction


def add_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add standard Tc/P/H bins and boolean regime columns."""
    out = frame.copy()
    true_tc = numeric_column(out, "Tc_true_K", default=float("nan"))
    if true_tc.isna().all() and "Tc_K" in out.columns:
        true_tc = numeric_column(out, "Tc_K", default=float("nan"))
    pressure = numeric_column(out, "pressure_GPa", default=0.0)
    field = numeric_column(out, "magnetic_field_T", default=0.0)
    formula = text_column(out, "formula_standardized")
    family = text_column(out, "family")
    fidelity = text_column(out, "fidelity")
    match_type = text_column(out, "match_type")

    out["tc_bin"] = pd.cut(true_tc, TC_BIN_EDGES, labels=TC_BIN_LABELS)
    out["pressure_bin"] = pd.cut(pressure, PRESSURE_BIN_EDGES, labels=PRESSURE_BIN_LABELS)
    out["field_bin"] = pd.cut(field, FIELD_BIN_EDGES, labels=FIELD_BIN_LABELS)
    out["is_h3s"] = formula.eq("H3S")
    out["is_ph3_sbh4"] = formula.isin(["PH3", "SbH4"])
    out["is_mgb2"] = family.eq("magnesium_boride") | formula.eq("MgB2")
    out["is_cuprate"] = family.eq("cuprate_or_cu_oxide")
    out["is_cuprate_high_tc"] = out["is_cuprate"] & (true_tc > 40.0)
    out["is_high_tc"] = true_tc > 40.0
    out["is_very_high_tc"] = true_tc > 90.0
    out["is_high_pressure"] = pressure > 50.0
    out["is_hydride_formula"] = formula.map(is_hydride_formula)
    out["is_hydride_high_pressure"] = out["is_hydride_formula"] & out["is_high_pressure"]
    out["is_formula_exact"] = match_type.eq("formula_exact")
    out["is_formula_similarity"] = match_type.eq("formula_similarity")
    out["is_exact_fidelity"] = fidelity.eq("exact")
    out["is_synthetic_doped"] = fidelity.eq("synthetic_doped")

    regime = pd.Series("ambient_common", index=out.index, dtype=object)
    regime.loc[out["is_high_tc"]] = "high_tc"
    regime.loc[out["is_very_high_tc"]] = "very_high_tc"
    regime.loc[out["is_high_pressure"]] = "high_pressure"
    regime.loc[out["is_hydride_high_pressure"]] = "hydride_high_pressure"
    regime.loc[out["is_cuprate"]] = "cuprate"
    regime.loc[out["is_mgb2"]] = "mgb2"
    out["regime"] = regime
    return out


def standard_subset_masks(frame: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Return standard subset masks for diagnostics."""
    enriched = add_regime_columns(frame)
    true_tc = numeric_column(enriched, "Tc_true_K", default=float("nan"))
    if true_tc.isna().all() and "Tc_K" in enriched.columns:
        true_tc = numeric_column(enriched, "Tc_K", default=float("nan"))
    return {
        "all": pd.Series(True, index=enriched.index),
        "exclude_H3S": ~enriched["is_h3s"],
        "exclude_PH3_SbH4": ~enriched["is_ph3_sbh4"],
        "exclude_hydride_high_pressure": ~enriched["is_hydride_high_pressure"],
        "exclude_Tc_gt_120": true_tc <= 120.0,
        "high_pressure_gt_50": enriched["is_high_pressure"],
        "high_pressure_gt_100": numeric_column(enriched, "pressure_GPa", default=0.0) > 100.0,
        "hydride_high_pressure": enriched["is_hydride_high_pressure"],
        "MgB2": enriched["is_mgb2"],
        "cuprate": enriched["is_cuprate"],
        "cuprate_high_tc": enriched["is_cuprate_high_tc"],
        "formula_exact": enriched["is_formula_exact"],
        "formula_similarity": enriched["is_formula_similarity"],
        "exact_fidelity": enriched["is_exact_fidelity"],
        "synthetic_doped": enriched["is_synthetic_doped"],
    }
