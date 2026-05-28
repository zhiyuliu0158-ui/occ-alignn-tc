"""Chemical formula parsing and fixed-size descriptors."""

from __future__ import annotations

import math
import re
import warnings
from typing import Mapping

import numpy as np
from pymatgen.core import Composition

from occ_alignn.featurizers.elements import NUM_ELEMENT_TOKENS, element_property_table, species_to_token

FORMULA_DESCRIPTOR_DIM = NUM_ELEMENT_TOKENS + 28
FORMULA_MISMATCH_DIM = 4
APPROX_FEATURE_DIM = 16
NODE_APPROX_FEATURE_DIM = 8


def parse_formula_counts(formula: object) -> dict[str, float]:
    """Parse a chemical formula into element counts.

    The parser normalizes common spacing variants such as ``Pb2 Te`` to the
    same count dictionary as ``Pb2Te``. Pymatgen handles parentheses and
    fractional stoichiometries, so the model receives one canonical vector.
    """
    if formula is None:
        return {}
    text = str(formula).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "unknown"}:
        return {}
    compact = re.sub(r"\s+", "", text)
    try:
        composition = Composition(compact)
    except Exception:
        warnings.warn(f"Failed to parse formula {formula!r}.", RuntimeWarning)
        return {}
    counts: dict[str, float] = {}
    for element, amount in composition.get_el_amt_dict().items():
        try:
            token = species_to_token(str(element))
        except Exception:
            continue
        if 0 < token < NUM_ELEMENT_TOKENS:
            counts[str(element)] = counts.get(str(element), 0.0) + float(amount)
    return counts


def counts_to_fraction_vector(counts: Mapping[str, float]) -> np.ndarray:
    """Return an element-fraction vector indexed by atomic number token."""
    vector = np.zeros(NUM_ELEMENT_TOKENS, dtype=np.float32)
    total = 0.0
    for symbol, amount in counts.items():
        value = float(amount)
        if value <= 0:
            continue
        try:
            token = species_to_token(symbol)
        except Exception:
            continue
        if 0 < token < NUM_ELEMENT_TOKENS:
            vector[token] += value
            total += value
    if total > 0:
        vector /= float(total)
    return vector


def formula_descriptor(formula: object) -> np.ndarray:
    """Return element fractions plus MAGPIE-like property statistics."""
    counts = parse_formula_counts(formula)
    fractions = counts_to_fraction_vector(counts)
    table = element_property_table()
    active = np.nonzero(fractions > 0)[0]
    descriptors = np.zeros(28, dtype=np.float32)
    if active.size == 0:
        descriptors[-1] = 1.0
        return np.concatenate([fractions, descriptors])

    weights = fractions[active]
    props = table[active]
    mean = (weights[:, None] * props).sum(axis=0)
    var = (weights[:, None] * (props - mean[None, :]) ** 2).sum(axis=0)
    prop_min = props.min(axis=0)
    prop_max = props.max(axis=0)
    total_atoms = float(sum(max(float(v), 0.0) for v in counts.values()))
    entropy = float(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum())
    descriptors[:6] = mean.astype(np.float32)
    descriptors[6:12] = var.astype(np.float32)
    descriptors[12:18] = prop_min.astype(np.float32)
    descriptors[18:24] = prop_max.astype(np.float32)
    descriptors[24] = math.log1p(total_atoms)
    descriptors[25] = entropy
    descriptors[26] = float(active.size) / 10.0
    descriptors[27] = 0.0
    return np.concatenate([fractions, descriptors.astype(np.float32)])


def formula_cif_mismatch(formula_features: np.ndarray, cif_formula: object) -> np.ndarray:
    """Compare row-level formula fractions with CIF-derived composition."""
    formula_frac = formula_features[:NUM_ELEMENT_TOKENS].astype(np.float32, copy=False)
    cif_frac = counts_to_fraction_vector(parse_formula_counts(cif_formula))
    diff = np.abs(formula_frac - cif_frac)
    formula_empty = float(formula_frac.sum() <= 0)
    cif_empty = float(cif_frac.sum() <= 0)
    return np.asarray(
        [
            float(diff.sum()),
            float(diff.max(initial=0.0)),
            formula_empty,
            cif_empty,
        ],
        dtype=np.float32,
    )
