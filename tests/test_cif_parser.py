"""Tests for CIF parsing and condition handling."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from occ_alignn.data.graph import parse_cif_sites
from occ_alignn.featurizers.elements import VACANCY_SYMBOL
from occ_alignn.featurizers.occupancy import parse_conditions


def _write_structure(structure: Structure, path) -> None:
    CifWriter(structure).write_file(str(path))


def test_full_cif_parses(tmp_path) -> None:
    path = tmp_path / "full.cif"
    structure = Structure(Lattice.cubic(3.5), ["Mg", "B"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    _write_structure(structure, path)
    _, coords, distributions, _ = parse_cif_sites(path, max_species_per_site=4)
    assert coords.shape == (2, 3)
    assert distributions[0]["Mg"] == 1.0
    assert distributions[1]["B"] == 1.0


def test_partial_cif_parses(tmp_path) -> None:
    path = tmp_path / "partial.cif"
    structure = Structure(
        Lattice.cubic(6.1),
        [{"Sn": 0.85, "Ag": 0.15}, "Te"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    _write_structure(structure, path)
    _, _, distributions, _ = parse_cif_sites(path, max_species_per_site=4)
    assert any("Sn" in dist and "Ag" in dist for dist in distributions)


def test_same_coord_species_merge(tmp_path) -> None:
    path = tmp_path / "merged.cif"
    path.write_text(
        """data_merge
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5
_cell_length_b   5
_cell_length_c   5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Sn Sn1 0 0 0 0.85
Ag Ag1 0 0 0 0.15
Te Te1 0.5 0.5 0.5 1.0
""",
        encoding="utf-8",
    )
    _, coords, distributions, _ = parse_cif_sites(path, max_species_per_site=4)
    assert coords.shape[0] == 2
    merged = [dist for dist in distributions if "Sn" in dist and "Ag" in dist][0]
    assert np.isclose(merged["Sn"], 0.85)
    assert np.isclose(merged["Ag"], 0.15)


def test_vacancy_added_for_under_occupied_site(tmp_path) -> None:
    path = tmp_path / "vacancy.cif"
    structure = Structure(Lattice.cubic(4.0), [{"O": 0.9}, "Cu"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    _write_structure(structure, path)
    _, _, distributions, _ = parse_cif_sites(path, max_species_per_site=4)
    vacancy_dist = [dist for dist in distributions if VACANCY_SYMBOL in dist][0]
    assert np.isclose(vacancy_dist[VACANCY_SYMBOL], 0.1)


def test_unknown_pressure_and_field() -> None:
    parsed = parse_conditions(
        {
            "pressure_GPa": "unknown",
            "magnetic_field_T": "",
            "field_direction": None,
        }
    )
    assert parsed.pressure.tolist() == [0.0, 0.0, 1.0]
    assert parsed.field.tolist() == [0.0, 0.0, 1.0]
    assert parsed.pressure_is_unknown == 1
    assert parsed.field_is_unknown == 1


def test_pressure_and_field_default_to_raw_log_features() -> None:
    parsed = parse_conditions(
        {
            "pressure_GPa": "170",
            "magnetic_field_T": "14",
            "field_direction": "parallel_c",
        }
    )

    assert np.isclose(parsed.pressure[0], 170.0)
    assert np.isclose(parsed.pressure[1], np.log1p(170.0))
    assert parsed.pressure[2] == 0.0
    assert np.isclose(parsed.field[0], 14.0)
    assert np.isclose(parsed.field[1], np.log1p(14.0))
    assert parsed.field[2] == 0.0


def test_pressure_and_field_can_use_log_scaled_features() -> None:
    parsed = parse_conditions(
        {
            "pressure_GPa": "170",
            "magnetic_field_T": "14",
            "field_direction": "parallel_c",
        },
        pressure_transform="log1p_scaled",
        field_transform="log1p_scaled",
    )

    assert np.isclose(parsed.pressure[0], np.log1p(170.0))
    assert np.isclose(parsed.pressure[1], np.log1p(170.0) / 6.0)
    assert np.isclose(parsed.field[0], np.log1p(14.0))
    assert np.isclose(parsed.field[1], np.log1p(14.0) / 4.0)
