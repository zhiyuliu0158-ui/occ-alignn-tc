"""Batching utilities for pure PyTorch crystal graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from occ_alignn.data.graph import GraphData


@dataclass
class GraphBatch:
    """A mini-batch of variable-size crystal graphs."""

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
    structure_features: torch.FloatTensor
    formula_features: torch.FloatTensor
    formula_mismatch_features: torch.FloatTensor
    confidence_features: torch.FloatTensor
    approx_features: torch.FloatTensor
    node_approx_features: torch.FloatTensor
    batch: torch.LongTensor
    edge_batch: torch.LongTensor
    angle_batch: torch.LongTensor
    pressure: torch.FloatTensor
    field: torch.FloatTensor
    field_direction_id: torch.LongTensor
    structure_source_id: torch.LongTensor
    match_type_id: torch.LongTensor
    fidelity_id: torch.LongTensor
    target: torch.FloatTensor
    tc_k: torch.FloatTensor
    sample_weight: torch.FloatTensor
    sample_ids: list[str]
    cif_paths: list[str]
    formulas: list[str]
    metadata: list[dict[str, object]]

    @property
    def num_graphs(self) -> int:
        """Return number of graphs in this batch."""
        return len(self.sample_ids)

    def to(self, device: torch.device | str) -> "GraphBatch":
        """Move tensor fields to a device."""
        tensor_fields = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if torch.is_tensor(value)
        }
        return GraphBatch(
            **tensor_fields,
            sample_ids=self.sample_ids,
            cif_paths=self.cif_paths,
            formulas=self.formulas,
            metadata=self.metadata,
        )


def _cat_or_empty(tensors: list[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if tensors:
        return torch.cat(tensors, dim=0)
    return torch.empty(shape, dtype=dtype)


def _cat_index_or_empty(tensors: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    if tensors:
        return torch.cat(tensors, dim=1)
    return torch.empty(shape, dtype=torch.long)


def collate_graphs(items: Iterable[GraphData]) -> GraphBatch:
    """Collate GraphData objects into one disconnected batch."""
    graphs = list(items)
    if not graphs:
        raise ValueError("Cannot collate an empty batch.")

    node_idx: list[torch.Tensor] = []
    node_occ: list[torch.Tensor] = []
    node_extra: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    edge_distance: list[torch.Tensor] = []
    edge_vector: list[torch.Tensor] = []
    edge_rbf: list[torch.Tensor] = []
    line_edge_index_list: list[torch.Tensor] = []
    angle: list[torch.Tensor] = []
    angle_rbf: list[torch.Tensor] = []
    angle_triplets: list[torch.Tensor] = []
    structure_features: list[torch.Tensor] = []
    formula_features: list[torch.Tensor] = []
    formula_mismatch_features: list[torch.Tensor] = []
    confidence_features: list[torch.Tensor] = []
    approx_features: list[torch.Tensor] = []
    node_approx: list[torch.Tensor] = []
    graph_batch: list[torch.Tensor] = []
    edge_batch: list[torch.Tensor] = []
    angle_batch: list[torch.Tensor] = []

    pressures: list[torch.Tensor] = []
    fields: list[torch.Tensor] = []
    field_direction_ids: list[int] = []
    structure_source_ids: list[int] = []
    match_type_ids: list[int] = []
    fidelity_ids: list[int] = []
    targets: list[float] = []
    tc_values: list[float] = []
    sample_weights: list[float] = []
    sample_ids: list[str] = []
    cif_paths: list[str] = []
    formulas: list[str] = []
    metadata: list[dict[str, object]] = []

    node_offset = 0
    edge_offset = 0
    for graph_id, graph in enumerate(graphs):
        node_idx.append(graph.node_species_idx)
        node_occ.append(graph.node_species_occ)
        node_extra.append(graph.node_extra_features)
        if graph.num_edges:
            edge_index_list.append(graph.edge_index + node_offset)
            edge_distance.append(graph.edge_distance)
            edge_vector.append(graph.edge_vector)
            edge_rbf.append(graph.edge_rbf)
            edge_batch.append(torch.full((graph.num_edges,), graph_id, dtype=torch.long))
        if graph.num_angles:
            line_edge_index_list.append(graph.line_edge_index + edge_offset)
            angle.append(graph.angle)
            angle_rbf.append(graph.angle_rbf)
            angle_triplets.append(graph.angle_triplet_index + node_offset)
            angle_batch.append(torch.full((graph.num_angles,), graph_id, dtype=torch.long))
        structure_features.append(graph.structure_features)
        formula_features.append(graph.formula_features)
        formula_mismatch_features.append(graph.formula_mismatch_features)
        confidence_features.append(graph.confidence_features)
        approx_features.append(graph.approx_features)
        node_approx.append(graph.node_approx_features)
        graph_batch.append(torch.full((graph.num_nodes,), graph_id, dtype=torch.long))
        pressures.append(graph.pressure if graph.pressure is not None else torch.zeros(3))
        fields.append(graph.field if graph.field is not None else torch.zeros(3))
        field_direction_ids.append(int(graph.field_direction_id))
        structure_source_ids.append(int(graph.structure_source_id))
        match_type_ids.append(int(graph.match_type_id))
        fidelity_ids.append(int(graph.fidelity_id))
        targets.append(float("nan") if graph.target is None else float(graph.target))
        tc_values.append(float("nan") if graph.tc_k is None else float(graph.tc_k))
        sample_weights.append(float(graph.sample_weight))
        sample_ids.append(graph.sample_id)
        cif_paths.append(graph.cif_path)
        formulas.append(graph.formula)
        metadata.append(graph.metadata or {})
        node_offset += graph.num_nodes
        edge_offset += graph.num_edges

    edge_rbf_dim = graphs[0].edge_rbf.shape[1]
    angle_rbf_dim = graphs[0].angle_rbf.shape[1]
    return GraphBatch(
        node_species_idx=torch.cat(node_idx, dim=0),
        node_species_occ=torch.cat(node_occ, dim=0),
        node_extra_features=torch.cat(node_extra, dim=0),
        edge_index=_cat_index_or_empty(edge_index_list, (2, 0)),
        edge_distance=_cat_or_empty(edge_distance, (0,), torch.float32),
        edge_vector=_cat_or_empty(edge_vector, (0, 3), torch.float32),
        edge_rbf=_cat_or_empty(edge_rbf, (0, edge_rbf_dim), torch.float32),
        line_edge_index=_cat_index_or_empty(line_edge_index_list, (2, 0)),
        angle=_cat_or_empty(angle, (0,), torch.float32),
        angle_rbf=_cat_or_empty(angle_rbf, (0, angle_rbf_dim), torch.float32),
        angle_triplet_index=_cat_index_or_empty(angle_triplets, (3, 0)),
        structure_features=torch.stack(structure_features).float(),
        formula_features=torch.stack(formula_features).float(),
        formula_mismatch_features=torch.stack(formula_mismatch_features).float(),
        confidence_features=torch.stack(confidence_features).float(),
        approx_features=torch.stack(approx_features).float(),
        node_approx_features=torch.cat(node_approx, dim=0).float(),
        batch=torch.cat(graph_batch, dim=0),
        edge_batch=_cat_or_empty(edge_batch, (0,), torch.long),
        angle_batch=_cat_or_empty(angle_batch, (0,), torch.long),
        pressure=torch.stack(pressures).float(),
        field=torch.stack(fields).float(),
        field_direction_id=torch.as_tensor(field_direction_ids, dtype=torch.long),
        structure_source_id=torch.as_tensor(structure_source_ids, dtype=torch.long),
        match_type_id=torch.as_tensor(match_type_ids, dtype=torch.long),
        fidelity_id=torch.as_tensor(fidelity_ids, dtype=torch.long),
        target=torch.as_tensor(targets, dtype=torch.float32),
        tc_k=torch.as_tensor(tc_values, dtype=torch.float32),
        sample_weight=torch.as_tensor(sample_weights, dtype=torch.float32),
        sample_ids=sample_ids,
        cif_paths=cif_paths,
        formulas=formulas,
        metadata=metadata,
    )
