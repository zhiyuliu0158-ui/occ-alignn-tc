"""Tests for graph batching offsets."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.graph import build_graph_from_cif


def _graph(path, species) :
    structure = Structure(Lattice.cubic(3.9), species, [[0, 0, 0], [0.5, 0.5, 0.5]])
    CifWriter(structure).write_file(str(path))
    graph = build_graph_from_cif(path, cutoff_radius=5.0, max_neighbors=6, n_edge_rbf=16, n_angle_rbf=8)
    graph.sample_id = str(path.stem)
    return graph


def test_two_graphs_batch_with_offsets(tmp_path) -> None:
    graph_a = _graph(tmp_path / "a.cif", ["Na", "Cl"])
    graph_b = _graph(tmp_path / "b.cif", ["Cs", "I"])
    batch = collate_graphs([graph_a, graph_b])
    assert batch.node_species_idx.shape[0] == graph_a.num_nodes + graph_b.num_nodes
    assert batch.edge_index.shape[1] == graph_a.num_edges + graph_b.num_edges
    assert batch.line_edge_index.shape[1] == graph_a.num_angles + graph_b.num_angles
    assert batch.edge_index.max().item() < batch.node_species_idx.shape[0]
    if batch.line_edge_index.numel():
        assert batch.line_edge_index.max().item() < batch.edge_index.shape[1]
    if batch.angle_triplet_index.numel():
        assert batch.angle_triplet_index.max().item() < batch.node_species_idx.shape[0]
    second_edges = batch.edge_index[:, batch.edge_batch == 1]
    assert second_edges.min().item() >= graph_a.num_nodes
