"""P-B-Occ-ALIGNN-full-Tc model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from occ_alignn.data.collate import GraphBatch
from occ_alignn.featurizers.elements import NUM_ELEMENT_TOKENS, PAD_TOKEN_ID
from occ_alignn.models.heads import GaussianTcHead
from occ_alignn.nn.film import ConditionEncoder, FiLM, mlp
from occ_alignn.nn.message_passing import ALIGNNLayer
from occ_alignn.nn.pooling import mean_max_pool, mean_pool


class PBOccALIGNNFullTc(nn.Module):
    """Occupancy-aware ALIGNN-style graph network for Tc regression."""

    node_extra_dim = 16
    disorder_dim = 4

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.element_emb_dim = int(model_cfg.get("element_emb_dim", 64))
        self.num_layers = int(model_cfg.get("num_layers", 4))
        self.n_edge_rbf = int(model_cfg.get("n_edge_rbf", 80))
        self.n_angle_rbf = int(model_cfg.get("n_angle_rbf", 40))
        self.use_explicit_triplet = bool(model_cfg.get("use_explicit_triplet", True))
        self.use_pressure_field_film = bool(model_cfg.get("use_pressure_field_film", True))
        self.use_max_pool = bool(model_cfg.get("use_max_pool", False))
        dropout = float(model_cfg.get("dropout", 0.1))

        self.element_embedding = nn.Embedding(
            NUM_ELEMENT_TOKENS,
            self.element_emb_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        self.node_projector = mlp(
            self.element_emb_dim + self.node_extra_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.pair_mlp = mlp(3 * self.element_emb_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.edge_mlp = mlp(
            self.n_edge_rbf + self.hidden_dim + 2 * self.disorder_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.triplet_mlp = mlp(4 * self.element_emb_dim, self.hidden_dim, self.hidden_dim, dropout)
        explicit_angle_dim = self.n_angle_rbf + self.hidden_dim + 3 * self.disorder_dim
        cheap_angle_dim = self.n_angle_rbf + 5 * self.hidden_dim
        self.angle_mlp = mlp(
            explicit_angle_dim if self.use_explicit_triplet else cheap_angle_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.condition_encoder = ConditionEncoder(self.hidden_dim, dropout)
        self.film_h = nn.ModuleList(FiLM(self.hidden_dim) for _ in range(self.num_layers))
        self.film_e = nn.ModuleList(FiLM(self.hidden_dim) for _ in range(self.num_layers))
        self.film_a = nn.ModuleList(FiLM(self.hidden_dim) for _ in range(self.num_layers))
        self.layers = nn.ModuleList(
            ALIGNNLayer(self.hidden_dim, dropout) for _ in range(self.num_layers)
        )
        pooled_dim = 2 * self.hidden_dim if self.use_max_pool else self.hidden_dim
        self.head = GaussianTcHead(pooled_dim + self.hidden_dim, self.hidden_dim, dropout)

    def _weighted_node_embedding(
        self,
        species_idx: torch.Tensor,
        species_occ: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.element_embedding(species_idx)
        return (species_occ.unsqueeze(-1) * emb).sum(dim=1)

    def _expected_pair_embedding(self, batch: GraphBatch) -> torch.Tensor:
        if batch.edge_index.numel() == 0:
            return torch.empty((0, self.hidden_dim), device=batch.node_species_idx.device)
        src = batch.edge_index[0]
        dst = batch.edge_index[1]
        idx_src = batch.node_species_idx[src]
        idx_dst = batch.node_species_idx[dst]
        occ_src = batch.node_species_occ[src]
        occ_dst = batch.node_species_occ[dst]
        emb_src = self.element_embedding(idx_src)
        emb_dst = self.element_embedding(idx_dst)
        output = torch.zeros((src.shape[0], self.hidden_dim), device=emb_src.device)
        for left in range(idx_src.shape[1]):
            for right in range(idx_dst.shape[1]):
                weight = occ_src[:, left] * occ_dst[:, right]
                features = torch.cat(
                    [
                        emb_src[:, left] + emb_dst[:, right],
                        torch.abs(emb_src[:, left] - emb_dst[:, right]),
                        emb_src[:, left] * emb_dst[:, right],
                    ],
                    dim=-1,
                )
                output = output + weight.unsqueeze(-1) * self.pair_mlp(features)
        return output

    def _expected_triplet_embedding(self, batch: GraphBatch) -> torch.Tensor:
        if batch.angle_triplet_index.numel() == 0:
            return torch.empty((0, self.hidden_dim), device=batch.node_species_idx.device)
        i = batch.angle_triplet_index[0]
        j = batch.angle_triplet_index[1]
        k = batch.angle_triplet_index[2]
        idx_i = batch.node_species_idx[i]
        idx_j = batch.node_species_idx[j]
        idx_k = batch.node_species_idx[k]
        occ_i = batch.node_species_occ[i]
        occ_j = batch.node_species_occ[j]
        occ_k = batch.node_species_occ[k]
        emb_i = self.element_embedding(idx_i)
        emb_j = self.element_embedding(idx_j)
        emb_k = self.element_embedding(idx_k)
        output = torch.zeros((i.shape[0], self.hidden_dim), device=emb_i.device)
        max_species = idx_i.shape[1]
        for left in range(max_species):
            for center in range(max_species):
                for right in range(max_species):
                    weight = occ_i[:, left] * occ_j[:, center] * occ_k[:, right]
                    features = torch.cat(
                        [
                            emb_j[:, center],
                            emb_i[:, left] + emb_k[:, right],
                            torch.abs(emb_i[:, left] - emb_k[:, right]),
                            emb_i[:, left] * emb_k[:, right],
                        ],
                        dim=-1,
                    )
                    output = output + weight.unsqueeze(-1) * self.triplet_mlp(features)
        return output

    def _initial_embeddings(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_emb = self._weighted_node_embedding(batch.node_species_idx, batch.node_species_occ)
        node_h = self.node_projector(torch.cat([node_emb, batch.node_extra_features], dim=-1))

        pair = self._expected_pair_embedding(batch)
        disorder = batch.node_extra_features[:, -self.disorder_dim :]
        if batch.edge_index.numel() == 0:
            edge_h = torch.empty((0, self.hidden_dim), device=node_h.device)
        else:
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            edge_features = torch.cat(
                [batch.edge_rbf, pair, disorder[src], disorder[dst]],
                dim=-1,
            )
            edge_h = self.edge_mlp(edge_features)

        if batch.angle_triplet_index.numel() == 0:
            angle_h = torch.empty((0, self.hidden_dim), device=node_h.device)
        elif self.use_explicit_triplet:
            i = batch.angle_triplet_index[0]
            j = batch.angle_triplet_index[1]
            k = batch.angle_triplet_index[2]
            triplet = self._expected_triplet_embedding(batch)
            angle_features = torch.cat(
                [batch.angle_rbf, triplet, disorder[i], disorder[j], disorder[k]],
                dim=-1,
            )
            angle_h = self.angle_mlp(angle_features)
        else:
            line_src = batch.line_edge_index[0]
            line_dst = batch.line_edge_index[1]
            center = batch.angle_triplet_index[1]
            e1 = edge_h[line_src]
            e2 = edge_h[line_dst]
            angle_features = torch.cat(
                [batch.angle_rbf, e1, e2, e1 * e2, torch.abs(e1 - e2), node_h[center]],
                dim=-1,
            )
            angle_h = self.angle_mlp(angle_features)
        return node_h, edge_h, angle_h

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        """Run a forward pass on a GraphBatch."""
        z_condition = self.condition_encoder(
            batch.pressure,
            batch.field,
            batch.field_direction_id,
            batch.structure_source_id,
            batch.match_type_id,
            batch.fidelity_id,
        )
        node_h, edge_h, angle_h = self._initial_embeddings(batch)
        for layer_id, layer in enumerate(self.layers):
            if self.use_pressure_field_film:
                node_h = self.film_h[layer_id](node_h, z_condition, batch.batch)
                edge_h = self.film_e[layer_id](edge_h, z_condition, batch.edge_batch)
                angle_h = self.film_a[layer_id](angle_h, z_condition, batch.angle_batch)
            node_h, edge_h, angle_h = layer(
                node_h=node_h,
                edge_h=edge_h,
                angle_h=angle_h,
                edge_index=batch.edge_index,
                line_edge_index=batch.line_edge_index,
            )
        pooled = (
            mean_max_pool(node_h, batch.batch, batch.num_graphs)
            if self.use_max_pool
            else mean_pool(node_h, batch.batch, batch.num_graphs)
        )
        final = torch.cat([pooled, z_condition], dim=-1)
        return self.head(final)
