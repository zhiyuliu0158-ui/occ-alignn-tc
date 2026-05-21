"""Partial occupancy site processing and condition parsing."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from occ_alignn.featurizers.elements import (
    PAD_TOKEN_ID,
    VACANCY_SYMBOL,
    element_property_table,
    species_to_token,
)

FIELD_DIRECTIONS = [
    "zero",
    "parallel_c",
    "parallel_ab",
    "parallel_a",
    "parallel_b",
    "powder",
    "unknown",
]
STRUCTURE_SOURCES = ["unknown", "MP", "COD", "ICSD", "other"]
MATCH_TYPES = ["unknown", "formula_exact", "doi_exact", "formula_similarity", "other"]
FIDELITIES = ["unknown", "exact", "synthetic_doped", "failed", "other"]


@dataclass(frozen=True)
class ConditionFeatures:
    """Numerical and categorical condition features for one sample."""

    pressure: np.ndarray
    field: np.ndarray
    field_direction_id: int
    structure_source_id: int
    match_type_id: int
    fidelity_id: int
    pressure_is_unknown: int
    field_is_unknown: int


def parse_float_with_unknown(value: object) -> tuple[float, int]:
    """Parse numeric values, using 0.0 plus unknown flag for missing values."""
    if value is None:
        return 0.0, 1
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "unknown", "null"}:
        return 0.0, 1
    try:
        out = float(text)
    except ValueError:
        return 0.0, 1
    if math.isnan(out) or math.isinf(out):
        return 0.0, 1
    return out, 0


def _category_id(value: object, vocab: list[str], default: str = "unknown") -> int:
    text = str(value).strip() if value is not None else default
    if text == "" or text.lower() in {"nan", "none", "null"}:
        text = default
    if text not in vocab:
        text = "other" if "other" in vocab else default
    return vocab.index(text)


def parse_conditions(row: Mapping[str, object]) -> ConditionFeatures:
    """Parse pressure, magnetic field, field direction, and metadata columns."""
    pressure_value, pressure_unknown = parse_float_with_unknown(row.get("pressure_GPa"))
    field_value, field_unknown = parse_float_with_unknown(row.get("magnetic_field_T"))
    direction = row.get("field_direction", "unknown")
    if direction is None or str(direction).strip() == "":
        direction = "unknown"
    pressure = np.asarray(
        [pressure_value, math.log1p(max(pressure_value, 0.0)), pressure_unknown],
        dtype=np.float32,
    )
    field = np.asarray(
        [field_value, math.log1p(max(field_value, 0.0)), field_unknown],
        dtype=np.float32,
    )
    return ConditionFeatures(
        pressure=pressure,
        field=field,
        field_direction_id=_category_id(direction, FIELD_DIRECTIONS),
        structure_source_id=_category_id(row.get("structure_source"), STRUCTURE_SOURCES),
        match_type_id=_category_id(row.get("match_type"), MATCH_TYPES),
        fidelity_id=_category_id(row.get("fidelity"), FIDELITIES),
        pressure_is_unknown=pressure_unknown,
        field_is_unknown=field_unknown,
    )


def normalize_site_distribution(
    species_occ: Mapping[str, float],
    max_species_per_site: int,
    occupancy_tol: float = 1e-5,
) -> dict[str, float]:
    """Normalize a site distribution and add Vacancy when occupancy is below one."""
    merged: dict[str, float] = {}
    for symbol, occ in species_occ.items():
        value = float(occ)
        if value <= 0:
            continue
        merged[symbol] = merged.get(symbol, 0.0) + value
    total = sum(merged.values())
    if total < 1.0 - occupancy_tol:
        merged[VACANCY_SYMBOL] = merged.get(VACANCY_SYMBOL, 0.0) + (1.0 - total)
        total = 1.0
    elif total <= 0:
        merged = {VACANCY_SYMBOL: 1.0}
        total = 1.0
    elif abs(total - 1.0) <= occupancy_tol:
        pass
    elif total > 1.0 + occupancy_tol:
        warnings.warn(
            f"Site occupancy sum {total:.6f} > 1; normalizing distribution.",
            RuntimeWarning,
        )
    if abs(total - 1.0) > 1e-12:
        merged = {symbol: occ / total for symbol, occ in merged.items()}
    if len(merged) > max_species_per_site:
        warnings.warn(
            f"Site has {len(merged)} species; keeping top {max_species_per_site}.",
            RuntimeWarning,
        )
        kept = sorted(merged.items(), key=lambda item: item[1], reverse=True)[
            :max_species_per_site
        ]
        kept_total = sum(occ for _, occ in kept)
        merged = {symbol: occ / kept_total for symbol, occ in kept}
    return merged


def distribution_to_tensors(
    distribution: Mapping[str, float], max_species_per_site: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a site distribution to fixed-size species id and occupancy arrays."""
    items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    idx = np.full(max_species_per_site, PAD_TOKEN_ID, dtype=np.int64)
    occ = np.zeros(max_species_per_site, dtype=np.float32)
    for pos, (symbol, value) in enumerate(items[:max_species_per_site]):
        idx[pos] = species_to_token(symbol)
        occ[pos] = float(value)
    return idx, occ


def site_extra_features(distribution: Mapping[str, float]) -> np.ndarray:
    """Return means, variances, and disorder descriptors for a distribution."""
    table = element_property_table()
    symbols = list(distribution.keys())
    probs = np.asarray([distribution[symbol] for symbol in symbols], dtype=np.float32)
    token_ids = np.asarray([species_to_token(symbol) for symbol in symbols], dtype=np.int64)
    props = table[token_ids]
    mean = (probs[:, None] * props).sum(axis=0)
    var = (probs[:, None] * (props - mean[None, :]) ** 2).sum(axis=0)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    max_occ = float(probs.max()) if len(probs) else 0.0
    n_species = float(len([symbol for symbol in symbols if symbol != VACANCY_SYMBOL]))
    vacancy_fraction = float(distribution.get(VACANCY_SYMBOL, 0.0))
    disorder = np.asarray([entropy, max_occ, n_species, vacancy_fraction], dtype=np.float32)
    return np.concatenate([mean.astype(np.float32), var.astype(np.float32), disorder])

