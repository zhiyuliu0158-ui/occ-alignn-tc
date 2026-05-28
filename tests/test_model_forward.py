"""Tests for model forward/backward."""

from __future__ import annotations

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.graph import build_graph_from_cif
from occ_alignn.models.occ_alignn_full_tc import PBOccALIGNNFullTc


def _config() -> dict:
    return {
        "model": {
            "hidden_dim": 32,
            "element_emb_dim": 24,
            "num_layers": 1,
            "dropout": 0.0,
            "max_species_per_site": 4,
            "cutoff_radius": 5.0,
            "max_neighbors": 6,
            "n_edge_rbf": 16,
            "n_angle_rbf": 8,
            "use_explicit_triplet": True,
        }
    }


def test_forward_backward(tmp_path) -> None:
    path = tmp_path / "model.cif"
    structure = Structure(Lattice.cubic(3.8), [{"Sn": 0.85, "Ag": 0.15}, "Te"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    CifWriter(structure).write_file(str(path))
    graph = build_graph_from_cif(path, cutoff_radius=5.0, max_neighbors=6, n_edge_rbf=16, n_angle_rbf=8)
    graph.target = torch.log1p(torch.tensor(4.2)).item()
    graph.tc_k = 4.2
    batch = collate_graphs([graph])
    model = PBOccALIGNNFullTc(_config())
    outputs = model(batch)
    assert set(outputs) == {"logtc_mu", "logtc_logvar", "Tc_pred_K"}
    assert outputs["logtc_mu"].shape == (1,)
    loss = (outputs["logtc_mu"] - batch.target).pow(2).mean()
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_confidence_branch_forward_backward(tmp_path) -> None:
    path = tmp_path / "confidence.cif"
    structure = Structure(Lattice.cubic(3.8), ["Mg", "B"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    CifWriter(structure).write_file(str(path))
    graph = build_graph_from_cif(path, cutoff_radius=5.0, max_neighbors=6, n_edge_rbf=16, n_angle_rbf=8)
    graph.confidence_features[3] = 1.0
    graph.target = torch.log1p(torch.tensor(39.0)).item()
    graph.tc_k = 39.0
    batch = collate_graphs([graph])
    config = _config()
    config["model"]["use_confidence_branch"] = True
    model = PBOccALIGNNFullTc(config)

    outputs = model(batch)
    loss = (outputs["logtc_mu"] - batch.target).pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.confidence_projector.parameters())
