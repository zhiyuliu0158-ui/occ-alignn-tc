"""Element tokens and elemental property tables."""

from __future__ import annotations

from functools import lru_cache
from typing import Final
import warnings

import numpy as np
from pymatgen.core.periodic_table import Element

PAD_TOKEN_ID: Final[int] = 0
VACANCY_TOKEN_ID: Final[int] = 119
NUM_ELEMENT_TOKENS: Final[int] = 120
VACANCY_SYMBOL: Final[str] = "Vacancy"


def species_to_token(symbol: str) -> int:
    """Map an element symbol or Vacancy to a token id."""
    if symbol == VACANCY_SYMBOL:
        return VACANCY_TOKEN_ID
    return int(Element(symbol).Z)


def token_to_species(token_id: int) -> str:
    """Map a token id to an element symbol or Vacancy."""
    if token_id == PAD_TOKEN_ID:
        return "PAD"
    if token_id == VACANCY_TOKEN_ID:
        return VACANCY_SYMBOL
    return Element.from_Z(int(token_id)).symbol


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return default
        return out
    except Exception:
        return default


def _quiet_getter(getter: object) -> object:
    """Evaluate a pymatgen property while suppressing missing-data warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return getter()
        except Exception:
            return None


def element_property_vector(symbol: str) -> np.ndarray:
    """Return [Z, mass, electronegativity, atomic_radius, group, period]."""
    if symbol == VACANCY_SYMBOL:
        return np.zeros(6, dtype=np.float32)
    el = Element(symbol)
    radius = _quiet_getter(lambda: el.atomic_radius) or _quiet_getter(
        lambda: el.atomic_radius_calculated
    )
    return np.asarray(
        [
            _safe_float(el.Z),
            _safe_float(el.atomic_mass),
            _safe_float(_quiet_getter(lambda: el.X)),
            _safe_float(radius),
            _safe_float(el.group),
            _safe_float(el.row),
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def element_property_table() -> np.ndarray:
    """Return property table indexed by token id."""
    table = np.zeros((NUM_ELEMENT_TOKENS, 6), dtype=np.float32)
    for z in range(1, 119):
        table[z] = element_property_vector(Element.from_Z(z).symbol)
    table[VACANCY_TOKEN_ID] = element_property_vector(VACANCY_SYMBOL)
    return table
