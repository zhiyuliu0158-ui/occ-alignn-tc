"""Tests for standard regime annotations and balanced sampler weights."""

from __future__ import annotations

import pandas as pd

from occ_alignn.analysis.regimes import add_regime_columns, standard_subset_masks
from occ_alignn.data.sampler import regime_sample_weights
from scripts.diagnose_data_distribution import _cif_reuse


def test_add_regime_columns_marks_key_subsets() -> None:
    df = pd.DataFrame(
        {
            "formula_standardized": ["H3S", "MgB2", "YBa2Cu3O7", "FeSe"],
            "family": ["other", "magnesium_boride", "cuprate_or_cu_oxide", "iron_based_or_fe_chalcogenide"],
            "Tc_K": [180.0, 39.0, 92.0, 9.0],
            "pressure_GPa": [150.0, 0.0, 0.0, 2.0],
            "match_type": ["formula_exact", "formula_exact", "formula_similarity", "formula_exact"],
            "fidelity": ["exact", "exact", "synthetic_doped", "exact"],
        }
    )

    out = add_regime_columns(df)

    assert out.loc[0, "is_hydride_high_pressure"]
    assert out.loc[1, "is_mgb2"]
    assert out.loc[2, "is_cuprate_high_tc"]
    assert out.loc[2, "is_formula_similarity"]
    assert out.loc[2, "is_synthetic_doped"]
    assert out.loc[0, "tc_bin"] == ">120"
    assert out.loc[0, "pressure_bin"] == ">100"


def test_standard_subset_masks_include_extreme_subsets() -> None:
    df = pd.DataFrame(
        {
            "formula_standardized": ["H3S", "PH3", "MgB2"],
            "family": ["other", "other", "magnesium_boride"],
            "Tc_true_K": [180.0, 100.0, 39.0],
            "pressure_GPa": [150.0, 200.0, 0.0],
            "match_type": ["formula_exact", "formula_exact", "formula_similarity"],
            "fidelity": ["exact", "exact", "synthetic_doped"],
        }
    )

    masks = standard_subset_masks(df)

    assert masks["hydride_high_pressure"].tolist() == [True, True, False]
    assert masks["exclude_H3S"].tolist() == [False, True, True]
    assert masks["MgB2"].tolist() == [False, False, True]
    assert masks["formula_similarity"].tolist() == [False, False, True]


def test_regime_sample_weights_are_capped() -> None:
    df = pd.DataFrame(
        {
            "formula_standardized": ["H3S", "YBa2Cu3O7", "FeSe"],
            "family": ["other", "cuprate_or_cu_oxide", "iron_based_or_fe_chalcogenide"],
            "Tc_K": [180.0, 92.0, 9.0],
            "pressure_GPa": [150.0, 0.0, 0.0],
            "match_type": ["formula_exact", "formula_similarity", "formula_exact"],
            "fidelity": ["exact", "synthetic_doped", "exact"],
        }
    )
    cfg = {
        "sampler_high_tc_weight": 1.5,
        "sampler_very_high_tc_weight": 2.0,
        "sampler_high_pressure_weight": 1.5,
        "sampler_hydride_high_pressure_weight": 1.5,
        "sampler_cuprate_weight": 1.5,
        "sampler_synthetic_doped_weight": 1.25,
        "sampler_max_weight": 4.0,
    }

    weights = regime_sample_weights(df, cfg).tolist()

    assert weights[0] == 4.0
    assert weights[1] == 4.0
    assert weights[2] == 1.0


def test_cif_reuse_handles_numeric_pressure_columns() -> None:
    df = pd.DataFrame(
        {
            "parent_cif_id": ["g1", "g1", "g2"],
            "formula_standardized": ["A", "A", "B"],
            "family": ["other", "other", "other"],
            "Tc_K": [1.0, 3.0, 5.0],
            "pressure_GPa": pd.to_numeric(["0", "10", "2"], errors="coerce"),
        }
    )

    reuse = _cif_reuse(df)

    row = reuse.loc[reuse["parent_cif_id"] == "g1"].iloc[0]
    assert row["tc_span"] == 2.0
    assert row["p_span"] == 10.0
