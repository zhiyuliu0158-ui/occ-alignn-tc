"""Create a tiny toy CIF dataset for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/toy")
    return parser.parse_args()


def _write_cif(structure: Structure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(structure).write_file(str(path))


def make_dataset(output_dir: str | Path) -> Path:
    """Write toy CIFs and toy_data.csv."""
    out = Path(output_dir)
    cif_dir = out / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    full = Structure(
        Lattice.cubic(3.52),
        ["Mg", "B", "B"],
        [[0, 0, 0], [1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]],
    )
    partial = Structure(
        Lattice.cubic(6.10),
        [{"Sn": 0.85, "Ag": 0.15}, "Te"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    vacancy = Structure(
        Lattice.cubic(4.15),
        [{"O": 0.90}, "Cu"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    cifs = {
        "toy_full.cif": full,
        "toy_partial.cif": partial,
        "toy_vacancy.cif": vacancy,
    }
    for filename, structure in cifs.items():
        _write_cif(structure, cif_dir / filename)

    rows = [
        {
            "sample_id": "toy_full_train",
            "cif_path": "cifs/toy_full.cif",
            "Tc_K": 39.0,
            "pressure_GPa": 0.0,
            "magnetic_field_T": 0.0,
            "field_direction": "zero",
            "structure_source": "toy",
            "match_type": "formula_exact",
            "fidelity": "exact",
            "parent_cif_id": "toy_full",
            "chemical_system": "B-Mg",
            "family": "toy",
            "split": "train",
        },
        {
            "sample_id": "toy_partial_train",
            "cif_path": "cifs/toy_partial.cif",
            "Tc_K": 4.2,
            "pressure_GPa": "unknown",
            "magnetic_field_T": "unknown",
            "field_direction": "unknown",
            "structure_source": "toy",
            "match_type": "formula_similarity",
            "fidelity": "synthetic_doped",
            "parent_cif_id": "toy_partial",
            "chemical_system": "Ag-Sn-Te",
            "family": "toy",
            "split": "train",
        },
        {
            "sample_id": "toy_vacancy_train",
            "cif_path": "cifs/toy_vacancy.cif",
            "Tc_K": 12.0,
            "pressure_GPa": 1.5,
            "magnetic_field_T": 3.0,
            "field_direction": "powder",
            "structure_source": "toy",
            "match_type": "formula_exact",
            "fidelity": "synthetic_doped",
            "parent_cif_id": "toy_vacancy",
            "chemical_system": "Cu-O",
            "family": "toy",
            "split": "train",
        },
        {
            "sample_id": "toy_full_val",
            "cif_path": "cifs/toy_full.cif",
            "Tc_K": 36.5,
            "pressure_GPa": 2.0,
            "magnetic_field_T": 0.0,
            "field_direction": "zero",
            "structure_source": "toy",
            "match_type": "formula_exact",
            "fidelity": "exact",
            "parent_cif_id": "toy_full_val",
            "chemical_system": "B-Mg",
            "family": "toy",
            "split": "val",
        },
        {
            "sample_id": "toy_partial_test",
            "cif_path": "cifs/toy_partial.cif",
            "Tc_K": 5.0,
            "pressure_GPa": "unknown",
            "magnetic_field_T": 1.0,
            "field_direction": "parallel_c",
            "structure_source": "toy",
            "match_type": "formula_similarity",
            "fidelity": "synthetic_doped",
            "parent_cif_id": "toy_partial_test",
            "chemical_system": "Ag-Sn-Te",
            "family": "toy",
            "split": "test",
        },
        {
            "sample_id": "toy_vacancy_test",
            "cif_path": "cifs/toy_vacancy.cif",
            "Tc_K": 10.0,
            "pressure_GPa": 0.0,
            "magnetic_field_T": "unknown",
            "field_direction": "unknown",
            "structure_source": "toy",
            "match_type": "formula_exact",
            "fidelity": "synthetic_doped",
            "parent_cif_id": "toy_vacancy_test",
            "chemical_system": "Cu-O",
            "family": "toy",
            "split": "test",
        },
    ]
    csv_path = out / "toy_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    args = parse_args()
    csv_path = make_dataset(args.output_dir)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
