"""Tests for periodic graph construction."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from occ_alignn.data.graph import build_graph_from_cif


def test_graph_tensors_and_indices(tmp_path) -> None:
    path = tmp_path / "graph.cif"
    structure = Structure(
        Lattice.cubic(3.8),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    CifWriter(structure).write_file(str(path))
    graph = build_graph_from_cif(path, cutoff_radius=5.0, max_neighbors=6, n_edge_rbf=16, n_angle_rbf=8)
    assert graph.node_species_idx.ndim == 2
    assert graph.node_species_occ.shape == graph.node_species_idx.shape
    assert graph.node_extra_features.shape[0] == graph.num_nodes
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_rbf.shape[0] == graph.num_edges
    assert graph.angle_rbf.shape[0] == graph.num_angles
    assert graph.angle_triplet_index.shape == (3, graph.num_angles)
    assert graph.edge_index.max().item() < graph.num_nodes
    if graph.num_angles:
        assert graph.line_edge_index.max().item() < graph.num_edges
        assert graph.angle_triplet_index.max().item() < graph.num_nodes
