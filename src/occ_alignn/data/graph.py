"""CIF parsing and periodic ALIGNN-style graph construction."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from occ_alignn.featurizers.occupancy import (
    distribution_to_tensors,
    normalize_site_distribution,
    site_extra_features,
)
from occ_alignn.nn.rbf import gaussian_rbf_np


@dataclass
class GraphData:
    """One crystal graph plus optional labels and metadata."""

    node_species_idx: torch.LongTensor
    node_species_occ: torch.FloatTensor
    node_extra_features: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_distance: torch.FloatTensor
    edge_vector: torch.FloatTensor
    edge_rbf: torch.FloatTensor
    line_edge_index: torch.LongTensor
    angle: torch.FloatTensor
    angle_rbf: torch.FloatTensor
    angle_triplet_index: torch.LongTensor
    pressure: torch.FloatTensor | None = None
    field: torch.FloatTensor | None = None
    field_direction_id: int = 6
    structure_source_id: int = 0
    match_type_id: int = 0
    fidelity_id: int = 0
    target: float | None = None
    tc_k: float | None = None
    sample_id: str = ""
    cif_path: str = ""
    formula: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def num_nodes(self) -> int:
        """Return number of crystallographic sites."""
        return int(self.node_species_idx.shape[0])

    @property
    def num_edges(self) -> int:
        """Return number of directed bonds."""
        return int(self.edge_index.shape[1])

    @property
    def num_angles(self) -> int:
        """Return number of line-graph angle edges."""
        return int(self.line_edge_index.shape[1])


def _specie_symbol(specie: object) -> str | None:
    """Return a clean element symbol for pymatgen Element/Species-like objects."""
    if hasattr(specie, "element"):
        symbol = getattr(specie.element, "symbol", None)
    else:
        symbol = getattr(specie, "symbol", None)
    if symbol is None:
        symbol = str(specie)
    try:
        return Element(str(symbol)).symbol
    except Exception:
        warnings.warn(f"Skipping unsupported species {specie!r}.", RuntimeWarning)
        return None


def _site_distribution(site: Any) -> dict[str, float]:
    """Extract species occupancy distribution from one pymatgen site."""
    distribution: dict[str, float] = {}
    try:
        species_items = site.species.items()
    except Exception:
        species_items = [(site.specie, 1.0)]
    for specie, occupancy in species_items:
        symbol = _specie_symbol(specie)
        if symbol is None:
            continue
        distribution[symbol] = distribution.get(symbol, 0.0) + float(occupancy)
    return distribution


def _periodic_coord_close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    delta = (a - b + 0.5) % 1.0 - 0.5
    return bool(np.max(np.abs(delta)) <= tol)


def parse_cif_sites(
    cif_path: str | Path,
    max_species_per_site: int,
    frac_coord_tol: float = 1e-4,
) -> tuple[Structure, np.ndarray, list[dict[str, float]], str]:
    """Parse CIF sites and merge same fractional coordinates into distributions.

    The returned sites are crystallographic sites. Same-position entries such as
    Sn(0.85) and Ag(0.15) are merged into one distribution instead of being
    expanded into an ordered supercell.
    """
    path = Path(cif_path)
    structure = Structure.from_file(str(path))
    merged_coords: list[np.ndarray] = []
    merged_distributions: list[dict[str, float]] = []

    for site in structure:
        frac = np.mod(np.asarray(site.frac_coords, dtype=np.float64), 1.0)
        frac[np.isclose(frac, 1.0, atol=frac_coord_tol)] = 0.0
        distribution = _site_distribution(site)
        if not distribution:
            warnings.warn(f"{path}: skipping site with no valid species.", RuntimeWarning)
            continue
        cluster_index = None
        for index, coord in enumerate(merged_coords):
            if _periodic_coord_close(frac, coord, frac_coord_tol):
                cluster_index = index
                break
        if cluster_index is None:
            merged_coords.append(frac)
            merged_distributions.append(distribution)
        else:
            for symbol, occupancy in distribution.items():
                merged_distributions[cluster_index][symbol] = (
                    merged_distributions[cluster_index].get(symbol, 0.0) + occupancy
                )

    if not merged_coords:
        raise ValueError(f"No valid sites parsed from CIF: {path}")

    normalized = [
        normalize_site_distribution(dist, max_species_per_site)
        for dist in merged_distributions
    ]
    formula_parts: list[str] = []
    for dist in normalized:
        formula_parts.extend(
            f"{symbol}{occupancy:.3g}" for symbol, occupancy in dist.items()
        )
    formula = " ".join(formula_parts)
    return structure, np.asarray(merged_coords, dtype=np.float64), normalized, formula


def _image_range(structure: Structure, cutoff_radius: float) -> range:
    lengths = np.asarray(structure.lattice.abc, dtype=np.float64)
    min_length = float(max(np.min(lengths), 1e-6))
    n_images = int(math.ceil(cutoff_radius / min_length)) + 1
    return range(-n_images, n_images + 1)


def _build_edges(
    structure: Structure,
    frac_coords: np.ndarray,
    cutoff_radius: float,
    max_neighbors: int,
    include_self_edges: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build directed periodic nearest-neighbor edges."""
    num_nodes = int(frac_coords.shape[0])
    edge_src: list[int] = []
    edge_dst: list[int] = []
    distances: list[float] = []
    vectors: list[np.ndarray] = []
    image_values = list(_image_range(structure, cutoff_radius))
    images = [
        np.asarray([ia, ib, ic], dtype=np.float64)
        for ia in image_values
        for ib in image_values
        for ic in image_values
    ]

    for src in range(num_nodes):
        candidates: list[tuple[float, int, np.ndarray]] = []
        for dst in range(num_nodes):
            for image in images:
                if not include_self_edges and src == dst and np.all(image == 0):
                    continue
                delta_frac = frac_coords[dst] + image - frac_coords[src]
                vector = structure.lattice.get_cartesian_coords(delta_frac)
                distance = float(np.linalg.norm(vector))
                if distance <= cutoff_radius and distance > 1e-8:
                    candidates.append((distance, dst, np.asarray(vector, dtype=np.float32)))
        candidates.sort(key=lambda item: item[0])
        for distance, dst, vector in candidates[:max_neighbors]:
            edge_src.append(src)
            edge_dst.append(dst)
            distances.append(distance)
            vectors.append(vector)

    if not edge_src:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
    return (
        edge_index,
        np.asarray(distances, dtype=np.float32),
        np.asarray(vectors, dtype=np.float32),
    )


def _build_angles(
    edge_index: np.ndarray,
    edge_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build line graph angle edges for directed bonds i->j and j->k."""
    num_edges = int(edge_index.shape[1])
    incoming_by_center: dict[int, list[int]] = {}
    outgoing_by_center: dict[int, list[int]] = {}
    for edge_id in range(num_edges):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        outgoing_by_center.setdefault(src, []).append(edge_id)
        incoming_by_center.setdefault(dst, []).append(edge_id)

    line_src: list[int] = []
    line_dst: list[int] = []
    angles: list[float] = []
    triplets: list[tuple[int, int, int]] = []
    for center, incoming_edges in incoming_by_center.items():
        for edge_in in incoming_edges:
            i = int(edge_index[0, edge_in])
            vec_in = -edge_vector[edge_in]
            for edge_out in outgoing_by_center.get(center, []):
                k = int(edge_index[1, edge_out])
                vec_out = edge_vector[edge_out]
                denom = float(np.linalg.norm(vec_in) * np.linalg.norm(vec_out))
                if denom <= 1e-12:
                    continue
                cosine = float(np.clip(np.dot(vec_in, vec_out) / denom, -1.0, 1.0))
                line_src.append(edge_in)
                line_dst.append(edge_out)
                angles.append(float(math.acos(cosine)))
                triplets.append((i, center, k))

    if not line_src:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((3, 0), dtype=np.int64),
        )
    return (
        np.asarray([line_src, line_dst], dtype=np.int64),
        np.asarray(angles, dtype=np.float32),
        np.asarray(triplets, dtype=np.int64).T,
    )


def build_graph_from_cif(
    cif_path: str | Path,
    max_species_per_site: int = 4,
    frac_coord_tol: float = 1e-4,
    cutoff_radius: float = 8.0,
    max_neighbors: int = 12,
    n_edge_rbf: int = 80,
    n_angle_rbf: int = 40,
    include_self_edges: bool = False,
) -> GraphData:
    """Build one occupancy-aware ALIGNN-style graph from a CIF file."""
    try:
        structure, frac_coords, distributions, formula = parse_cif_sites(
            cif_path=cif_path,
            max_species_per_site=max_species_per_site,
            frac_coord_tol=frac_coord_tol,
        )
    except Exception as exc:
        warnings.warn(f"Failed to parse CIF {cif_path}: {exc}", RuntimeWarning)
        raise

    species_idx: list[np.ndarray] = []
    species_occ: list[np.ndarray] = []
    extra_features: list[np.ndarray] = []
    for distribution in distributions:
        idx, occ = distribution_to_tensors(distribution, max_species_per_site)
        species_idx.append(idx)
        species_occ.append(occ)
        extra_features.append(site_extra_features(distribution))

    edge_index, edge_distance, edge_vector = _build_edges(
        structure=structure,
        frac_coords=frac_coords,
        cutoff_radius=cutoff_radius,
        max_neighbors=max_neighbors,
        include_self_edges=include_self_edges,
    )
    edge_rbf = gaussian_rbf_np(edge_distance, 0.0, cutoff_radius, n_edge_rbf)
    line_edge_index, angles, angle_triplet_index = _build_angles(edge_index, edge_vector)
    angle_rbf = gaussian_rbf_np(angles, 0.0, math.pi, n_angle_rbf)

    return GraphData(
        node_species_idx=torch.as_tensor(np.stack(species_idx), dtype=torch.long),
        node_species_occ=torch.as_tensor(np.stack(species_occ), dtype=torch.float32),
        node_extra_features=torch.as_tensor(np.stack(extra_features), dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_distance=torch.as_tensor(edge_distance, dtype=torch.float32),
        edge_vector=torch.as_tensor(edge_vector, dtype=torch.float32),
        edge_rbf=torch.as_tensor(edge_rbf, dtype=torch.float32),
        line_edge_index=torch.as_tensor(line_edge_index, dtype=torch.long),
        angle=torch.as_tensor(angles, dtype=torch.float32),
        angle_rbf=torch.as_tensor(angle_rbf, dtype=torch.float32),
        angle_triplet_index=torch.as_tensor(angle_triplet_index, dtype=torch.long),
        cif_path=str(cif_path),
        formula=formula,
    )
