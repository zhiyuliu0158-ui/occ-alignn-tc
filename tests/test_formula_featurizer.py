"""Tests for row-level chemical formula descriptors."""

from __future__ import annotations

import numpy as np

from occ_alignn.featurizers.elements import species_to_token
from occ_alignn.featurizers.formula import (
    FORMULA_DESCRIPTOR_DIM,
    counts_to_fraction_vector,
    formula_cif_mismatch,
    formula_descriptor,
    parse_formula_counts,
)


def test_dataset_uses_row_formula_instead_of_cif_formula(tmp_path) -> None:
    import pandas as pd
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    from occ_alignn.data.dataset import CifTcDataset

    path = tmp_path / "la2cuo4.cif"
    structure = Structure(
        Lattice.tetragonal(3.8, 13.2),
        ["Cu", "La", "La", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0, 0, 0.35],
            [0, 0, 0.65],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0.2],
            [0.5, 0.5, 0.8],
        ],
    )
    CifWriter(structure).write_file(str(path))
    df = pd.DataFrame(
        [
            {
                "sample_id": "doped",
                "cif_path": str(path),
                "formula_standardized": "CuLa1.95Nd0.05O4",
                "Tc_K": 30.0,
            }
        ]
    )
    config = {
        "model": {"max_species_per_site": 4, "apply_virtual_formula_doping": False},
        "data": {},
    }

    graph = CifTcDataset(df, config, csv_dir=tmp_path)[0]
    nd_token = species_to_token("Nd")

    assert np.isclose(graph.formula_features[nd_token].item(), 0.05 / 7.0)
    assert graph.node_species_occ[graph.node_species_idx == nd_token].sum().item() == 0.0
    assert graph.formula_mismatch_features[0].item() > 0.0


def test_virtual_doping_projects_formula_to_host_site(tmp_path) -> None:
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    from occ_alignn.data.graph import build_graph_from_cif

    path = tmp_path / "la2cuo4.cif"
    structure = Structure(
        Lattice.tetragonal(3.8, 13.2),
        ["Cu", "La", "La", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0, 0, 0.35],
            [0, 0, 0.65],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0.2],
            [0.5, 0.5, 0.8],
        ],
    )
    CifWriter(structure).write_file(str(path))

    graph = build_graph_from_cif(
        path,
        max_species_per_site=4,
        target_formula="CuLa1.95Nd0.05O4",
        apply_virtual_doping=True,
    )
    nd_token = species_to_token("Nd")
    la_token = species_to_token("La")
    nd_occ = graph.node_species_occ[graph.node_species_idx == nd_token].sum().item()
    la_occ = graph.node_species_occ[graph.node_species_idx == la_token].sum().item()

    assert nd_occ > 0.03
    assert la_occ < 1.98


def test_parse_formula_compacts_spacing() -> None:
    assert parse_formula_counts("Pb2Te") == parse_formula_counts("Pb2 Te")


def test_parse_formula_handles_parentheses_and_decimals() -> None:
    counts = parse_formula_counts("Ba(Fe0.5Co0.5)2As2")
    assert counts["Ba"] == 1.0
    assert counts["Fe"] == 1.0
    assert counts["Co"] == 1.0
    assert counts["As"] == 2.0


def test_fraction_vector_uses_atomic_number_tokens() -> None:
    vector = counts_to_fraction_vector({"Pb": 2.0, "Te": 1.0})
    assert np.isclose(vector[species_to_token("Pb")], 2.0 / 3.0)
    assert np.isclose(vector[species_to_token("Te")], 1.0 / 3.0)
    assert np.isclose(vector.sum(), 1.0)


def test_formula_descriptor_and_mismatch_shapes() -> None:
    descriptor = formula_descriptor("Pb2Te")
    mismatch = formula_cif_mismatch(descriptor, "Pb2Te")
    assert descriptor.shape == (FORMULA_DESCRIPTOR_DIM,)
    assert mismatch.shape == (4,)
    assert np.isclose(mismatch[0], 0.0)
