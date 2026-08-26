#!/usr/bin/env python3
"""3DSC-style XGBoost evaluation on the final_cleanned package.

This follows the 3DSC ML protocol as closely as possible on the local data:
  - MAGPIE composition features from ChemML's magpie_python generators.
  - DSOAP structure features with 3DSC SOAP hyperparameters.
  - Crystal temperature, symmetry one-hot, and lattice constants for the
    MAGPIE+DSOAP setting.
  - GroupShuffleSplit by chemical system (`chemical_composition_sc`).
  - XGBRegressor() with sample weights and metric weights.
  - 3DSC target transform: restricted_arcsinh(Tc), inverse restricted_sinh.

Additional cleaned experimental-condition features are appended:
pressure_GPa, magnetic_field_T, is_high_pressure, Tc_criterion,
substrate_formula. Raw fields and synthesis_method are not used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_mp_exact_cifs import FormulaError, parse_formula, write_csv_xlsx


SOAP_R_CUT = 4.270
SOAP_N_MAX = 6
SOAP_L_MAX = 4
SOAP_SIGMA = 0.336
SOAP_CROSSOVER = False
SYM_FEATURES = [
    "cubic",
    "hexagonal",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "triclinic",
    "trigonal",
    "primitive",
    "base-centered",
    "body-centered",
    "face-centered",
]
LATTICE_FEATURES = ["lata_2", "latb_2", "latc_2"]
STRUCTURE_AUX_FEATURES = ["crystal_temp_2"] + SYM_FEATURES + LATTICE_FEATURES
ELEMENT_ALIASES = {
    "D": "H",
    "T": "H",
}
CONDITION_NUMERIC_COLUMNS = [
    "pressure_GPa",
    "magnetic_field_T",
    "is_high_pressure",
]
CONDITION_CATEGORICAL_COLUMNS = [
    "Tc_criterion",
    "substrate_formula",
]
EXCLUDED_INPUT_COLUMNS = [
    "pressure_raw",
    "magnetic_field_raw",
    "criterion_raw",
    "synthesis_method",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official 3DSC-style XGB evaluation.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            r"D:\xwechat_files\wxid_xrz1o1l56kbf22_41da\msg\file\2026-06\final_cleanned\final_cleanned"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("official_3dsc_xgb_with_conditions"),
    )
    parser.add_argument(
        "--datasets",
        default="exact_cif_only,exact_plus_3dsc",
        help="Comma-separated dataset folder names.",
    )
    parser.add_argument(
        "--feature-sets",
        default="MAGPIE,MAGPIE+DSOAP",
        help="Comma-separated feature sets to evaluate.",
    )
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=58)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--soap-n-jobs", type=int, default=1)
    parser.add_argument(
        "--exclude-high-pressure",
        action="store_true",
        help="Optional: drop rows marked is_high_pressure before evaluation.",
    )
    parser.add_argument(
        "--max-pressure-gpa",
        type=float,
        default=None,
        help="Optional: keep only rows with pressure_GPa <= this value.",
    )
    parser.add_argument(
        "--skip-dsoap",
        action="store_true",
        help="Debug fallback: skip DSOAP even when MAGPIE+DSOAP is requested.",
    )
    return parser.parse_args()


def restricted_arcsinh(x: np.ndarray) -> np.ndarray:
    norm = np.arcsinh(1 / 2) * 10
    return np.arcsinh(np.maximum(x, 0) / 2) / norm


def restricted_sinh(x: np.ndarray) -> np.ndarray:
    norm = np.arcsinh(1 / 2) * 10
    y = x * norm
    y = np.clip(y, np.arcsinh(0), np.arcsinh(200 / 2))
    return np.sinh(y) * 2


def weighted_average(values: np.ndarray, weights: np.ndarray | None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def metrics_3dsc(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score

    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    y_true = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    out = {
        "MSLE": float(mean_squared_log_error(y_true, y_pred, sample_weight=weights)),
        "MAE": float(mean_absolute_error(y_true, y_pred, sample_weight=weights)),
        "r2": float(r2_score(y_true, y_pred, sample_weight=weights)),
    }
    err = y_pred - y_true
    out["RMSE"] = float(math.sqrt(weighted_average(err**2, weights)))
    out["median_AE"] = float(np.median(np.abs(err)))
    return out


def boolean_array(series: pd.Series) -> np.ndarray:
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"}).to_numpy(dtype=bool)


def composition_to_formula(comp: dict[str, Any]) -> str:
    def amount_to_text(value: Any) -> str:
        value = float(value)
        if abs(value - round(value)) < 1e-10:
            n = int(round(value))
            return "" if n == 1 else str(n)
        return (f"{value:.8f}").rstrip("0").rstrip(".")

    normalized: dict[str, float] = {}
    for element, amount in comp.items():
        element = ELEMENT_ALIASES.get(str(element), str(element))
        normalized[element] = normalized.get(element, 0.0) + float(amount)
    return "".join(f"{el}{amount_to_text(amount)}" for el, amount in sorted(normalized.items()))


def formula_elements(formula: Any) -> list[str]:
    comp = parse_formula(str(formula))
    return sorted({ELEMENT_ALIASES.get(element, element) for element in comp.keys()})


def chemical_system(formula: Any) -> str:
    return "-".join(formula_elements(formula))


def clean_formula_for_magpie(row: pd.Series) -> str:
    for col in ["formula_standardized", "formula_reduced"]:
        value = str(row.get(col, "")).strip()
        if not value or value.lower() == "nan":
            continue
        try:
            comp = parse_formula(value)
            return composition_to_formula(comp)
        except FormulaError:
            continue
    return str(row.get("formula_standardized", row.get("formula_reduced", ""))).strip()


def build_cif_index(dataset_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for folder_name in ["train_cif_files", "test_cif_files", "cif_files"]:
        folder = dataset_root / folder_name
        if not folder.exists():
            continue
        for cif_path in folder.rglob("*.cif"):
            if folder_name == "cif_files":
                rel = (Path("cif_files") / cif_path.relative_to(folder)).as_posix()
            else:
                rel = cif_path.relative_to(folder).as_posix()
            index.setdefault(rel, cif_path)
    return index


def resolve_cif_paths(df: pd.DataFrame, dataset_root: Path) -> tuple[list[Path], int]:
    cif_index = build_cif_index(dataset_root)
    paths: list[Path] = []
    missing = 0
    for _, row in df.iterrows():
        rel = str(row.get("primary_cif_relpath", "")).replace("\\", "/").strip()
        path = cif_index.get(rel)
        if path is None:
            candidate = dataset_root / rel
            path = candidate if candidate.exists() else None
        if path is None:
            missing += 1
            path = Path("")
        paths.append(path)
    return paths, missing


def install_fake_chemml_package() -> None:
    import chemml as chemml_real

    site = Path(chemml_real.__file__).resolve().parent
    fake_chemml = types.ModuleType("chemml")
    fake_chemml.__path__ = [str(site)]
    fake_chem = types.ModuleType("chemml.chem")
    fake_chem.__path__ = [str(site / "chem")]
    sys.modules["chemml"] = fake_chemml
    sys.modules["chemml.chem"] = fake_chem


def load_magpie_classes():
    import numpy as np_local

    for alias, value in [("float", float), ("int", int), ("bool", bool)]:
        if not hasattr(np_local, alias):
            setattr(np_local, alias, value)
    install_fake_chemml_package()
    from chemml.chem.magpie_python import (
        CompositionEntry,
        ElementalPropertyAttributeGenerator,
        IonicityAttributeGenerator,
        StoichiometricAttributeGenerator,
        ValenceShellAttributeGenerator,
    )

    return (
        CompositionEntry,
        ValenceShellAttributeGenerator,
        ElementalPropertyAttributeGenerator,
        StoichiometricAttributeGenerator,
        IonicityAttributeGenerator,
    )


def calculate_magpie_features(formulas: pd.Series, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    (
        CompositionEntry,
        ValenceShellAttributeGenerator,
        ElementalPropertyAttributeGenerator,
        StoichiometricAttributeGenerator,
        IonicityAttributeGenerator,
    ) = load_magpie_classes()

    entries = []
    used_formulas: list[str] = []
    for formula in formulas.astype(str):
        comp = CompositionEntry()
        comp_dict = comp.parse_composition(formula)
        comp.set_composition(amounts=comp_dict.values(), element_ids=comp_dict.keys())
        entries.append(comp)
        used_formulas.append(formula)

    magpie_features: pd.DataFrame | None = None
    generators = [
        ValenceShellAttributeGenerator(),
        ElementalPropertyAttributeGenerator(),
        StoichiometricAttributeGenerator(),
        IonicityAttributeGenerator(),
    ]
    for generator in generators:
        features = generator.generate_features(entries=entries)
        magpie_features = features if magpie_features is None else magpie_features.join(features)

    assert magpie_features is not None
    not_nan_cols = ~magpie_features.isna().any(axis=0)
    magpie_features = magpie_features.loc[:, not_nan_cols]
    magpie_features.columns = [f"MAGPIE.{col}" for col in magpie_features.columns]
    magpie_features.insert(0, "formula_sc", used_formulas)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    magpie_features.to_pickle(cache_path)
    return magpie_features


def elements_in_structure(structure: Any) -> list[str]:
    out = []
    for site in structure:
        for species in site.species:
            try:
                out.append(ELEMENT_ALIASES.get(species.element.symbol, species.element.symbol))
            except AttributeError:
                out.append(ELEMENT_ALIASES.get(species.symbol, species.symbol))
    return sorted(set(out))


class DisorderedSOAPCompat:
    def __init__(self, species: list[str], symprec: float = 0.01):
        from dscribe.descriptors import SOAP

        self.symprec = symprec
        compression = {"mode": "crossover"} if SOAP_CROSSOVER is False else {"mode": "off"}
        self.soap = SOAP(
            r_cut=SOAP_R_CUT,
            n_max=SOAP_N_MAX,
            l_max=SOAP_L_MAX,
            sigma=SOAP_SIGMA,
            species=species,
            sparse=False,
            periodic=True,
            dtype="float32",
            average="off",
            compression=compression,
        )

    @staticmethod
    def total_occupancy(site: Any) -> float:
        return float(sum(site.species.as_dict().values()))

    @staticmethod
    def normalize_isotope_species(struct: Any) -> Any:
        from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation

        replacements: dict[int, Any] = {}
        for i, site in enumerate(struct):
            species_and_occu = site.species.as_dict()
            normalized: dict[str, float] = {}
            changed = False
            for species, occu in species_and_occu.items():
                new_species = ELEMENT_ALIASES.get(str(species), str(species))
                changed = changed or new_species != str(species)
                normalized[new_species] = normalized.get(new_species, 0.0) + float(occu)
            if changed:
                replacements[i] = normalized if len(normalized) > 1 else next(iter(normalized))
        if replacements:
            struct = ReplaceSiteSpeciesTransformation(replacements).apply_transformation(struct)
        return struct

    @staticmethod
    def get_all_atoms(ordered_struct: Any) -> list[str]:
        atoms = []
        for species in ordered_struct.species:
            try:
                atoms.append(species.element.name)
            except AttributeError:
                atoms.append(species.name)
        return atoms

    def normalise_total_occupancy_of_site(self, struct: Any) -> tuple[Any, list[float]]:
        from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation

        site_weights: list[float] = []
        upscale: dict[int, Any] = {}
        eps = 1e-8
        for i, site in enumerate(struct):
            species_and_occu = site.species.as_dict()
            total = sum(species_and_occu.values())
            site_weights.append(total)
            if total > 1 + eps:
                warnings.warn(f"Total occupancy > 1: {total}")
            if total != 1:
                if len(species_and_occu) == 1:
                    upscale[i] = list(species_and_occu.keys())[0]
                elif len(species_and_occu) > 1:
                    factor = 1 / total
                    upscale[i] = {species: occu * factor for species, occu in species_and_occu.items()}
        if upscale:
            struct = ReplaceSiteSpeciesTransformation(upscale).apply_transformation(struct)
        return struct, site_weights

    def get_disordered_sites(self, symm_struct: Any) -> dict[str, list[Any]]:
        sites = [equiv_sites[0] for equiv_sites in symm_struct.equivalent_sites]
        all_indices = symm_struct.equivalent_indices
        disordered_sites = {key: [] for key in ["elements", "occupancies", "indices"]}
        for indices, site in zip(all_indices, sites):
            if len(site.species) > 1:
                disordered_sites["elements"].append(list(site.species.as_dict().keys()))
                disordered_sites["occupancies"].append(list(site.species.as_dict().values()))
                disordered_sites["indices"].append(indices)
        return disordered_sites

    def get_ordered_ase_structures_and_weights(self, pymatgen_struct: Any) -> tuple[list[Any], list[float]]:
        from itertools import product

        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation

        symm_struct = SpacegroupAnalyzer(pymatgen_struct, symprec=self.symprec).get_symmetrized_structure()
        disordered_sites = self.get_disordered_sites(symm_struct)

        if len(disordered_sites["elements"]) == 0:
            return [AseAtomsAdaptor().get_atoms(pymatgen_struct)], [1.0]

        if not np.allclose([sum(occus) for occus in disordered_sites["occupancies"]], 1):
            warnings.warn("Disordered occupancies do not sum to 1; using normalized combination weights.")

        ase_structures: list[Any] = []
        doping_weights: list[float] = []
        element_combs = product(*disordered_sites["elements"])
        occupancy_combs = product(*disordered_sites["occupancies"])
        for all_elements, all_occupancies in zip(element_combs, occupancy_combs):
            weight = float(np.prod(all_occupancies))
            replace_sites: dict[int, Any] = {}
            for element, indices in zip(all_elements, disordered_sites["indices"]):
                for idx in indices:
                    replace_sites[idx] = element
            ordered_struct = ReplaceSiteSpeciesTransformation(replace_sites).apply_transformation(pymatgen_struct)
            ase_struct = AseAtomsAdaptor().get_atoms(ordered_struct)
            if self.get_all_atoms(ordered_struct) != ase_struct.get_chemical_symbols():
                warnings.warn("ASE and pymatgen atom orders differ for a DSOAP structure.")
            ase_structures.append(ase_struct)
            doping_weights.append(weight)
        return ase_structures, doping_weights

    @staticmethod
    def weighted_average(array: np.ndarray, weights: list[float], axis: int = 0) -> np.ndarray:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.sum() <= 0:
            weights_array = np.ones_like(weights_array)
        return np.average(array, axis=axis, weights=weights_array).astype(np.float32)

    def get_soap_vector(self, raw_pymatgen_struct: Any) -> np.ndarray:
        raw_pymatgen_struct = self.normalize_isotope_species(raw_pymatgen_struct)
        pymatgen_struct, site_weights = self.normalise_total_occupancy_of_site(raw_pymatgen_struct)
        ase_structures, doping_weights = self.get_ordered_ase_structures_and_weights(pymatgen_struct)
        site_features = self.soap.create(ase_structures, n_jobs=1)
        site_features = np.asarray(site_features, dtype=np.float32)
        if site_features.ndim == 2:
            site_features = site_features[None, :, :]
        structure_features = self.weighted_average(site_features, site_weights, axis=1)
        return self.weighted_average(structure_features, doping_weights, axis=0)

    def get_number_of_features(self) -> int:
        return int(self.soap.get_number_of_features())


def structure_aux_features(structure: Any) -> dict[str, float]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    values = {name: 0.0 for name in STRUCTURE_AUX_FEATURES}
    values["crystal_temp_2"] = 0.0
    values["lata_2"] = float(structure.lattice.a)
    values["latb_2"] = float(structure.lattice.b)
    values["latc_2"] = float(structure.lattice.c)
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
        crystal_system = str(analyzer.get_crystal_system()).lower()
        if crystal_system in values:
            values[crystal_system] = 1.0
        symbol = analyzer.get_space_group_symbol().strip().upper()
        centering = symbol[:1]
        if centering == "P":
            values["primitive"] = 1.0
        elif centering in {"A", "B", "C"}:
            values["base-centered"] = 1.0
        elif centering == "I":
            values["body-centered"] = 1.0
        elif centering == "F":
            values["face-centered"] = 1.0
    except Exception:
        pass
    return values


def load_structures(unique_paths: list[Path]) -> tuple[dict[str, Any], dict[str, str]]:
    from pymatgen.core import Structure

    structures: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for path in unique_paths:
        key = str(path)
        try:
            structures[key] = Structure.from_file(path)
        except Exception as exc:
            errors[key] = repr(exc)
    return structures, errors


def calculate_structure_features(
    paths: list[Path],
    formulas: pd.Series,
    cache_npz: Path,
    cache_meta: Path,
    soap_n_jobs: int,
    skip_dsoap: bool,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    rel_keys = [str(path) for path in paths]
    unique_paths = sorted({path for path in paths if str(path)})
    if cache_npz.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        cached = np.load(cache_npz, allow_pickle=False)
        unique_keys = list(cached["unique_keys"].astype(str))
        matrix = cached["features"].astype(np.float32)
        aux = pd.DataFrame(cached["aux"], columns=list(cached["aux_names"].astype(str)))
        feature_names = list(cached["feature_names"].astype(str))
        row_by_key = {key: i for i, key in enumerate(unique_keys)}
        row_matrix = np.zeros((len(paths), matrix.shape[1]), dtype=np.float32)
        row_aux = np.zeros((len(paths), aux.shape[1]), dtype=np.float32)
        for i, key in enumerate(rel_keys):
            idx = row_by_key.get(key)
            if idx is not None:
                row_matrix[i, :] = matrix[idx, :]
                row_aux[i, :] = aux.to_numpy(dtype=np.float32)[idx, :]
        return row_matrix, feature_names, pd.DataFrame(row_aux, columns=aux.columns), meta

    structures, parse_errors = load_structures(unique_paths)
    all_elements: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        try:
            elems = formula_elements(formula)
        except Exception:
            elems = []
        for element in elems:
            if element not in seen:
                seen.add(element)
                all_elements.append(element)
    for structure in structures.values():
        for element in elements_in_structure(structure):
            if element not in seen:
                seen.add(element)
                all_elements.append(element)

    soap_matrix_unique: np.ndarray
    soap_names: list[str]
    if skip_dsoap:
        soap_matrix_unique = np.zeros((len(unique_paths), 0), dtype=np.float32)
        soap_names = []
        soap_meta = {"skipped": True}
    else:
        soap = DisorderedSOAPCompat(species=all_elements)
        n_features = soap.get_number_of_features()
        soap_matrix_unique = np.zeros((len(unique_paths), n_features), dtype=np.float32)
        soap_names = [f"SOAP_{i}" for i in range(n_features)]
        soap_errors: dict[str, str] = {}
        for i, path in enumerate(unique_paths):
            key = str(path)
            structure = structures.get(key)
            if structure is None:
                continue
            try:
                soap_matrix_unique[i, :] = soap.get_soap_vector(structure)
            except Exception as exc:
                soap_errors[key] = repr(exc)
            if (i + 1) % 100 == 0:
                print(f"  DSOAP {i + 1}/{len(unique_paths)}")
        soap_meta = {
            "skipped": False,
            "r_cut": SOAP_R_CUT,
            "n_max": SOAP_N_MAX,
            "l_max": SOAP_L_MAX,
            "sigma": SOAP_SIGMA,
            "crossover": SOAP_CROSSOVER,
            "species": all_elements,
            "n_features": n_features,
            "soap_errors": soap_errors,
        }

    aux_rows: list[dict[str, float]] = []
    for path in unique_paths:
        structure = structures.get(str(path))
        if structure is None:
            aux_rows.append({name: 0.0 for name in STRUCTURE_AUX_FEATURES})
        else:
            aux_rows.append(structure_aux_features(structure))
    aux_unique = pd.DataFrame(aux_rows, columns=STRUCTURE_AUX_FEATURES).astype(np.float32)

    row_by_key = {str(path): i for i, path in enumerate(unique_paths)}
    row_soap = np.zeros((len(paths), soap_matrix_unique.shape[1]), dtype=np.float32)
    row_aux = np.zeros((len(paths), aux_unique.shape[1]), dtype=np.float32)
    aux_array = aux_unique.to_numpy(dtype=np.float32)
    for i, key in enumerate(rel_keys):
        idx = row_by_key.get(key)
        if idx is not None:
            row_soap[i, :] = soap_matrix_unique[idx, :]
            row_aux[i, :] = aux_array[idx, :]

    meta = {
        "unique_cifs": len(unique_paths),
        "structure_parse_errors": parse_errors,
        "soap": soap_meta,
        "aux_features": STRUCTURE_AUX_FEATURES,
    }
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        unique_keys=np.asarray([str(path) for path in unique_paths]),
        features=soap_matrix_unique,
        feature_names=np.asarray(soap_names),
        aux=aux_array,
        aux_names=np.asarray(STRUCTURE_AUX_FEATURES),
    )
    cache_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return row_soap, soap_names, pd.DataFrame(row_aux, columns=STRUCTURE_AUX_FEATURES), meta


def prepare_dataset(
    spec: DatasetSpec,
    output_dir: Path,
    soap_n_jobs: int,
    exclude_high_pressure: bool,
    max_pressure_gpa: float | None,
    skip_dsoap: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(spec.root / "new_extracted_data_with_split.csv")
    df = df[pd.to_numeric(df["Tc_K"], errors="coerce").notna()].reset_index(drop=True)
    rows_before_filter = int(len(df))
    keep = np.ones(len(df), dtype=bool)
    if exclude_high_pressure:
        keep &= ~boolean_array(df["is_high_pressure"])
    if max_pressure_gpa is not None:
        pressure = pd.to_numeric(df["pressure_GPa"], errors="coerce").fillna(0.0).to_numpy()
        keep &= pressure <= max_pressure_gpa
    df = df.loc[keep].reset_index(drop=True)

    df["formula_sc"] = df.apply(clean_formula_for_magpie, axis=1)
    df["tc"] = pd.to_numeric(df["Tc_K"], errors="coerce").astype(float)
    df["chemical_composition_sc"] = df["formula_sc"].map(chemical_system)
    df["weight"] = 1.0 / df.groupby("formula_sc")["formula_sc"].transform("size")

    paths, missing_cif_rows = resolve_cif_paths(df, spec.root)
    df["resolved_cif_path"] = [str(path) for path in paths]

    dataset_cache = output_dir / spec.name / "feature_cache"
    magpie_unique = calculate_magpie_features(
        pd.Series(sorted(df["formula_sc"].unique())),
        dataset_cache / "magpie_by_formula.pkl",
    )
    magpie_features = magpie_unique.set_index("formula_sc").loc[df["formula_sc"]].reset_index(drop=True)

    soap_matrix, soap_names, aux_features, structure_meta = calculate_structure_features(
        paths=paths,
        formulas=df["formula_sc"],
        cache_npz=dataset_cache / f"structure_features_skip{int(skip_dsoap)}.npz",
        cache_meta=dataset_cache / f"structure_features_skip{int(skip_dsoap)}.json",
        soap_n_jobs=soap_n_jobs,
        skip_dsoap=skip_dsoap,
    )

    condition = pd.DataFrame(index=df.index)
    for col in CONDITION_NUMERIC_COLUMNS:
        if col == "is_high_pressure":
            condition[f"COND.{col}"] = boolean_array(df[col]).astype(float)
        else:
            condition[f"COND.{col}"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in CONDITION_CATEGORICAL_COLUMNS:
        values = df[col].fillna("<empty>").astype(str).str.strip().replace({"": "<empty>"})
        dummies = pd.get_dummies(values, prefix=f"COND.{col}", dtype=float)
        condition = condition.join(dummies)

    out = df.copy()
    out = pd.concat([out, magpie_features, aux_features, condition], axis=1)
    if soap_matrix.shape[1] > 0:
        soap_df = pd.DataFrame(soap_matrix, columns=soap_names)
        out = pd.concat([out, soap_df], axis=1)

    manifest = {
        "rows_before_filter": rows_before_filter,
        "rows_after_filter": int(len(out)),
        "rows_removed_by_filter": int(rows_before_filter - len(out)),
        "missing_cif_rows": int(missing_cif_rows),
        "unique_formula_sc": int(out["formula_sc"].nunique()),
        "unique_chemical_composition_sc": int(out["chemical_composition_sc"].nunique()),
        "n_magpie_features": int(len([c for c in out.columns if c.startswith("MAGPIE.")])),
        "n_soap_features": int(len(soap_names)),
        "n_condition_features": int(condition.shape[1]),
        "structure_meta": structure_meta,
    }
    return out, manifest


def make_group_splits(df: pd.DataFrame, n_reps: int, train_frac: float, random_seed: int) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(train_size=train_frac, n_splits=n_reps, random_state=random_seed)
    groups = df["chemical_composition_sc"].to_numpy()
    split_df = pd.DataFrame(index=df.index)
    x_dummy = np.zeros((len(df), 1))
    for rep, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, df["tc"], groups=groups)):
        col = f"CV_{rep}"
        values = np.full(len(df), "", dtype=object)
        values[train_idx] = "train"
        values[test_idx] = "test"
        split_df[col] = values
    return split_df


def train_one_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    output_root: Path,
    n_reps: int,
    train_frac: float,
    random_seed: int,
    xgb_n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from xgboost import XGBRegressor

    feature_cols: list[str] = []
    magpie_cols = [col for col in df.columns if col.startswith("MAGPIE.")]
    soap_cols = [col for col in df.columns if col.startswith("SOAP_")]
    condition_cols = [col for col in df.columns if col.startswith("COND.")]
    if "MAGPIE" in feature_set:
        feature_cols += magpie_cols
    if "DSOAP" in feature_set:
        feature_cols += soap_cols + STRUCTURE_AUX_FEATURES
    feature_cols += condition_cols

    if "DSOAP" in feature_set and not soap_cols:
        raise RuntimeError("MAGPIE+DSOAP requested but no SOAP_ features are available.")
    if not feature_cols:
        raise RuntimeError(f"No features selected for {feature_set}")

    cv_cols = [f"CV_{i}" for i in range(n_reps)]
    y = df["tc"].to_numpy(dtype=float)
    weights = df["weight"].to_numpy(dtype=float)
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    predictions_all = df[
        [
            "record_id",
            "formula_standardized",
            "formula_reduced",
            "formula_sc",
            "chemical_composition_sc",
            "tc",
            "pressure_GPa",
            "magnetic_field_T",
            "Tc_criterion",
            "substrate_formula",
            "cif_match_type",
            "primary_cif_relpath",
            "resolved_cif_path",
            "weight",
        ]
    ].copy()
    for col in cv_cols:
        predictions_all[col] = df[col].to_numpy()

    score_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for rep, cv_col in enumerate(cv_cols):
        train_mask = df[cv_col].eq("train").to_numpy()
        test_mask = df[cv_col].eq("test").to_numpy()
        train_groups = set(df.loc[train_mask, "chemical_composition_sc"])
        test_groups = set(df.loc[test_mask, "chemical_composition_sc"])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"chemical system leakage in rep {rep}: {sorted(overlap)[:5]}")

        x_train, x_test = x_all[train_mask], x_all[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        w_train, w_test = weights[train_mask], weights[test_mask]

        model = XGBRegressor(n_jobs=xgb_n_jobs, random_state=random_seed)
        regressor = Pipeline([("StandardScaler", StandardScaler()), ("model", model)])
        target_transformer = FunctionTransformer(
            func=restricted_arcsinh,
            inverse_func=restricted_sinh,
            check_inverse=False,
        )
        regr = TransformedTargetRegressor(
            regressor=regressor,
            transformer=target_transformer,
        )
        regr.fit(x_train, y_train, model__sample_weight=w_train)
        pred_train = np.maximum(regr.predict(x_train), 0.0)
        pred_test = np.maximum(regr.predict(x_test), 0.0)
        pred_all = np.maximum(regr.predict(x_all), 0.0)
        predictions_all[f"XGB_{rep}_tc_pred"] = pred_all

        train_scores = metrics_3dsc(y_train, pred_train, w_train)
        test_scores = metrics_3dsc(y_test, pred_test, w_test)
        for split_name, scores, n_rows, n_groups in [
            ("train", train_scores, int(train_mask.sum()), len(train_groups)),
            ("test", test_scores, int(test_mask.sum()), len(test_groups)),
        ]:
            row = {
                "features": feature_set,
                "model": "XGB",
                "repetition": rep,
                "split": split_name,
                "n_rows": n_rows,
                "n_chemical_systems": int(n_groups),
            }
            row.update(scores)
            score_rows.append(row)

        split_rows.append(
            {
                "features": feature_set,
                "repetition": rep,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_chemical_systems": int(len(train_groups)),
                "test_chemical_systems": int(len(test_groups)),
                "chemical_system_overlap": int(len(overlap)),
                "train_frac_rows": float(train_mask.mean()),
                "test_frac_rows": float(test_mask.mean()),
            }
        )
        print(
            f"    {feature_set} rep={rep}: test MSLE={test_scores['MSLE']:.4f}, "
            f"MAE={test_scores['MAE']:.3f}, R2={test_scores['r2']:.3f}",
            flush=True,
        )

    scores_by_split = pd.DataFrame(score_rows)
    split_summary = pd.DataFrame(split_rows)
    feature_safe = feature_set.replace("+", "_")
    write_csv_xlsx(scores_by_split, output_root / f"{feature_safe}_scores_by_split.csv")
    write_csv_xlsx(split_summary, output_root / f"{feature_safe}_split_summary.csv")
    write_csv_xlsx(predictions_all, output_root / f"{feature_safe}_all_values_and_predictions.csv")

    manifest = {
        "feature_set": feature_set,
        "n_features": int(len(feature_cols)),
        "n_magpie_features": int(len([c for c in feature_cols if c.startswith("MAGPIE.")])),
        "n_soap_features": int(len([c for c in feature_cols if c.startswith("SOAP_")])),
        "n_structure_aux_features": int(len([c for c in feature_cols if c in STRUCTURE_AUX_FEATURES])),
        "n_condition_features": int(len([c for c in feature_cols if c.startswith("COND.")])),
        "feature_columns": feature_cols,
    }
    return scores_by_split, split_summary, manifest


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["MSLE", "MAE", "RMSE", "median_AE", "r2", "n_rows", "n_chemical_systems"]
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
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    manifest: dict[str, Any] = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "protocol": "3DSC-style XGB: GroupShuffleSplit by chemical_composition_sc, train_frac=0.8, n_reps=100 by default, sample/metric weights by formula_sc, arcsinh target transform.",
        "reference_repo": "https://github.com/aimat-lab/3DSC",
        "reference_paper": "https://www.nature.com/articles/s41597-023-02721-y",
        "random_seed": args.random_seed,
        "n_reps": args.n_reps,
        "train_frac": args.train_frac,
        "feature_sets": feature_sets,
        "condition_numeric_columns": CONDITION_NUMERIC_COLUMNS,
        "condition_categorical_columns_one_hot": CONDITION_CATEGORICAL_COLUMNS,
        "excluded_input_columns": EXCLUDED_INPUT_COLUMNS,
        "exclude_high_pressure": bool(args.exclude_high_pressure),
        "max_pressure_gpa": args.max_pressure_gpa,
        "python_executable": sys.executable,
    }

    all_scores: list[pd.DataFrame] = []
    for dataset_name in dataset_names:
        spec = DatasetSpec(dataset_name, base_dir / dataset_name)
        dataset_output = output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {dataset_name} ===")
        df, dataset_manifest = prepare_dataset(
            spec=spec,
            output_dir=output_dir,
            soap_n_jobs=args.soap_n_jobs,
            exclude_high_pressure=args.exclude_high_pressure,
            max_pressure_gpa=args.max_pressure_gpa,
            skip_dsoap=args.skip_dsoap,
        )
        split_df = make_group_splits(df, args.n_reps, args.train_frac, args.random_seed)
        df = pd.concat([df, split_df], axis=1)
        write_csv_xlsx(
            df[
                [
                    "record_id",
                    "formula_sc",
                    "chemical_composition_sc",
                    "tc",
                    "pressure_GPa",
                    "magnetic_field_T",
                    "Tc_criterion",
                    "substrate_formula",
                    "weight",
                ]
                + [f"CV_{i}" for i in range(args.n_reps)]
            ],
            dataset_output / "3dsc_group_splits.csv",
        )

        dataset_manifest["feature_sets"] = {}
        for feature_set in feature_sets:
            print(f"  Training {feature_set}")
            scores, _split_summary, feature_manifest = train_one_feature_set(
                df=df,
                feature_set=feature_set,
                output_root=dataset_output,
                n_reps=args.n_reps,
                train_frac=args.train_frac,
                random_seed=args.random_seed,
                xgb_n_jobs=args.xgb_n_jobs,
            )
            scores.insert(0, "dataset", dataset_name)
            all_scores.append(scores)
            dataset_manifest["feature_sets"][feature_set] = feature_manifest
        manifest[dataset_name] = dataset_manifest

    all_scores_df = pd.concat(all_scores, ignore_index=True)
    aggregate = aggregate_scores(all_scores_df)
    write_csv_xlsx(all_scores_df, output_dir / "summary_scores_by_split.csv")
    write_csv_xlsx(aggregate, output_dir / "summary_scores_aggregate.csv")
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nAggregate test scores:")
    view = aggregate[aggregate["split"] == "test"][
        [
            "dataset",
            "features",
            "n_reps",
            "MSLE_mean",
            "MSLE_std",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
            "r2_mean",
            "r2_std",
        ]
    ]
    print(view.to_string(index=False))
    print(f"\noutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
