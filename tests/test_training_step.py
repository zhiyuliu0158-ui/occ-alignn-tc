"""Tests for one optimization step."""

from __future__ import annotations

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.graph import build_graph_from_cif
from occ_alignn.models.occ_alignn_full_tc import PBOccALIGNNFullTc
from occ_alignn.training.losses import regression_loss


def test_one_training_step(tmp_path) -> None:
    path = tmp_path / "train.cif"
    structure = Structure(Lattice.cubic(3.8), ["Mg", "B"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    CifWriter(structure).write_file(str(path))
    graph = build_graph_from_cif(path, cutoff_radius=5.0, max_neighbors=6, n_edge_rbf=16, n_angle_rbf=8)
    graph.target = torch.log1p(torch.tensor(39.0)).item()
    graph.tc_k = 39.0
    batch = collate_graphs([graph])
    config = {
        "model": {
            "hidden_dim": 32,
            "element_emb_dim": 24,
            "num_layers": 1,
            "dropout": 0.0,
            "max_species_per_site": 4,
            "n_edge_rbf": 16,
            "n_angle_rbf": 8,
            "use_explicit_triplet": True,
        },
        "training": {"regression_loss": "gaussian_nll"},
    }
    model = PBOccALIGNNFullTc(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    loss = regression_loss(outputs, batch.target, "gaussian_nll")
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_mixed_log_raw_huber_loss_backward() -> None:
    outputs = {
        "logtc_mu": torch.tensor([torch.log1p(torch.tensor(40.0)).item()], requires_grad=True),
        "logtc_logvar": torch.zeros(1),
    }
    target = torch.log1p(torch.tensor([180.0]))

    loss = regression_loss(
        outputs,
        target,
        "mixed_log_raw_huber",
        raw_weight=0.15,
        raw_scale=50.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs["logtc_mu"].grad is not None


def test_weighted_huber_loss_changes_gradient() -> None:
    outputs = {
        "logtc_mu": torch.tensor([1.0, 1.0], requires_grad=True),
        "logtc_logvar": torch.zeros(2),
    }
    target = torch.tensor([1.0, 3.0])
    weight = torch.tensor([1.0, 4.0])

    loss = regression_loss(outputs, target, "huber", weight=weight)
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs["logtc_mu"].grad is not None
    assert abs(outputs["logtc_mu"].grad[1]) > abs(outputs["logtc_mu"].grad[0])
