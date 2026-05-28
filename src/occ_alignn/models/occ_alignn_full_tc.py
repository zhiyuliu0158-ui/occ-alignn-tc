"""P-B-Occ-ALIGNN-full-Tc model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from occ_alignn.data.collate import GraphBatch
from occ_alignn.featurizers.elements import NUM_ELEMENT_TOKENS, PAD_TOKEN_ID
from occ_alignn.featurizers.formula import (
    APPROX_FEATURE_DIM,
    FORMULA_DESCRIPTOR_DIM,
    FORMULA_MISMATCH_DIM,
    NODE_APPROX_FEATURE_DIM,
)
from occ_alignn.models.heads import GaussianTcHead
from occ_alignn.nn.film import ConditionEncoder, FiLM, mlp
from occ_alignn.nn.message_passing import ALIGNNLayer
from occ_alignn.nn.pooling import mean_max_pool, mean_pool


class PBOccALIGNNFullTc(nn.Module):
    """Occupancy-aware ALIGNN-style graph network for Tc regression."""

    node_extra_dim = 16
    disorder_dim = 4
    confidence_dim = 8

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
        self.use_composition_branch = bool(model_cfg.get("use_composition_branch", True))
        self.use_formula_branch = bool(model_cfg.get("use_formula_branch", True))
        self.use_structure_branch = bool(model_cfg.get("use_structure_branch", True))
        self.use_mismatch_branch = bool(model_cfg.get("use_mismatch_branch", True))
        self.use_confidence_branch = bool(model_cfg.get("use_confidence_branch", False))
        self.use_attention_pool = bool(model_cfg.get("use_attention_pool", True))
        self.use_approximation_branch = bool(model_cfg.get("use_approximation_branch", False))
        dropout = float(model_cfg.get("dropout", 0.1))

        self.element_embedding = nn.Embedding(
            NUM_ELEMENT_TOKENS,
            self.element_emb_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        node_input_dim = 3 * self.element_emb_dim + self.node_extra_dim
        if self.use_approximation_branch:
            node_input_dim += NODE_APPROX_FEATURE_DIM
        self.node_projector = mlp(
            node_input_dim,
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
        self.attention_score = mlp(3 * self.hidden_dim, self.hidden_dim, 1, dropout)
        graph_input_dim = pooled_dim + (self.hidden_dim if self.use_attention_pool else 0)
        self.graph_projector = mlp(graph_input_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.composition_projector = mlp(
            NUM_ELEMENT_TOKENS + 2 * self.element_emb_dim + self.node_extra_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.structure_projector = mlp(8, self.hidden_dim, self.hidden_dim, dropout)
        self.formula_projector = mlp(
            FORMULA_DESCRIPTOR_DIM,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.mismatch_projector = mlp(
            FORMULA_MISMATCH_DIM,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.confidence_projector = mlp(
            self.confidence_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.approx_projector = mlp(
            APPROX_FEATURE_DIM,
            self.hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.material_projector = mlp(6 * self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.fusion_gate = mlp(8 * self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.head = GaussianTcHead(10 * self.hidden_dim, self.hidden_dim, dropout)

    def _weighted_node_embedding(
        self,
        species_idx: torch.Tensor,
        species_occ: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.element_embedding(species_idx)
        return (species_occ.unsqueeze(-1) * emb).sum(dim=1)

    def _site_species_embedding(
        self,
        species_idx: torch.Tensor,
        species_occ: torch.Tensor,
    ) -> torch.Tensor:
        """Represent mixed sites without hiding low-occupancy dopants.

        The old weighted mean is still included, but majority and minority
        species get explicit channels. This lets a La0.975/Nd0.025 site expose
        an Nd signal instead of changing the La embedding by only 2.5%.
        """
        emb = self.element_embedding(species_idx)
        valid = (species_idx != PAD_TOKEN_ID) & (species_occ > 0)
        weighted = (species_occ.unsqueeze(-1) * emb).sum(dim=1)
        majority = emb[:, 0]
        minority_mask = valid.clone()
        minority_mask[:, 0] = False
        minority_count = minority_mask.sum(dim=1, keepdim=True).clamp_min(1)
        minority = (minority_mask.unsqueeze(-1).to(emb.dtype) * emb).sum(dim=1)
        minority = minority / minority_count.to(emb.dtype)
        minority = torch.where(minority_mask.any(dim=1, keepdim=True), minority, torch.zeros_like(minority))
        return torch.cat([weighted, majority, minority], dim=-1)

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
        node_emb = self._site_species_embedding(batch.node_species_idx, batch.node_species_occ)
        node_features = [node_emb, batch.node_extra_features]
        if self.use_approximation_branch:
            node_features.append(batch.node_approx_features)
        node_h = self.node_projector(torch.cat(node_features, dim=-1))

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

    def _composition_embedding(self, batch: GraphBatch) -> torch.Tensor:
        """Build graph-level composition features from partially occupied sites."""
        species_idx = batch.node_species_idx
        species_occ = batch.node_species_occ
        valid = species_idx != PAD_TOKEN_ID
        flat_idx = species_idx.reshape(-1)
        flat_occ = species_occ.reshape(-1) * valid.reshape(-1).to(species_occ.dtype)
        flat_graph = batch.batch.unsqueeze(1).expand_as(species_idx).reshape(-1)

        counts = torch.zeros(
            (batch.num_graphs, NUM_ELEMENT_TOKENS),
            dtype=species_occ.dtype,
            device=species_occ.device,
        )
        counts.index_put_((flat_graph, flat_idx), flat_occ, accumulate=True)
        counts[:, PAD_TOKEN_ID] = 0.0
        counts = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        site_emb = self._weighted_node_embedding(species_idx, species_occ)
        emb_sum = torch.zeros(
            (batch.num_graphs, self.element_emb_dim),
            dtype=site_emb.dtype,
            device=site_emb.device,
        )
        emb_sum.index_add_(0, batch.batch, site_emb)
        graph_sizes = torch.bincount(batch.batch, minlength=batch.num_graphs).to(site_emb.dtype)
        emb_mean = emb_sum / graph_sizes.unsqueeze(-1).clamp_min(1.0)

        emb_max = torch.full_like(emb_sum, -torch.inf)
        extra_sum = torch.zeros(
            (batch.num_graphs, self.node_extra_dim),
            dtype=batch.node_extra_features.dtype,
            device=batch.node_extra_features.device,
        )
        extra_sum.index_add_(0, batch.batch, batch.node_extra_features)
        for graph_id in range(batch.num_graphs):
            mask = batch.batch == graph_id
            if bool(mask.any()):
                emb_max[graph_id] = site_emb[mask].max(dim=0).values
        emb_max = torch.where(torch.isfinite(emb_max), emb_max, torch.zeros_like(emb_max))
        extra_mean = extra_sum / graph_sizes.unsqueeze(-1).clamp_min(1.0)

        features = torch.cat([counts, emb_mean, emb_max, extra_mean], dim=-1)
        if not self.use_composition_branch:
            return torch.zeros(
                (batch.num_graphs, self.hidden_dim),
                dtype=features.dtype,
                device=features.device,
            )
        return self.composition_projector(features)

    def _attention_pool(
        self,
        node_h: torch.Tensor,
        comp_repr: torch.Tensor,
        z_condition: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Condition-aware node attention pooling."""
        context = torch.cat([node_h, comp_repr[batch_index], z_condition[batch_index]], dim=-1)
        scores = self.attention_score(context).squeeze(-1)
        pooled = torch.zeros((num_graphs, node_h.shape[-1]), dtype=node_h.dtype, device=node_h.device)
        for graph_id in range(num_graphs):
            mask = batch_index == graph_id
            if bool(mask.any()):
                weights = torch.softmax(scores[mask], dim=0)
                pooled[graph_id] = (weights.unsqueeze(-1) * node_h[mask]).sum(dim=0)
        return pooled

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
        comp_repr = self._composition_embedding(batch)
        struct_repr = self.structure_projector(batch.structure_features)
        formula_repr = self.formula_projector(batch.formula_features)
        mismatch_repr = self.mismatch_projector(batch.formula_mismatch_features)
        confidence_repr = self.confidence_projector(batch.confidence_features)
        approx_repr = self.approx_projector(batch.approx_features)
        if not self.use_structure_branch:
            struct_repr = torch.zeros_like(struct_repr)
        if not self.use_formula_branch:
            formula_repr = torch.zeros_like(formula_repr)
        if not self.use_mismatch_branch:
            mismatch_repr = torch.zeros_like(mismatch_repr)
        if not self.use_confidence_branch:
            confidence_repr = torch.zeros_like(confidence_repr)
        if not self.use_approximation_branch:
            approx_repr = torch.zeros_like(approx_repr)
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
        if self.use_attention_pool:
            attn_pooled = self._attention_pool(
                node_h,
                comp_repr,
                z_condition,
                batch.batch,
                batch.num_graphs,
            )
            graph_input = torch.cat([pooled, attn_pooled], dim=-1)
        else:
            graph_input = pooled
        graph_repr = self.graph_projector(graph_input)
        material_repr = self.material_projector(
            torch.cat(
                [formula_repr, comp_repr, struct_repr, mismatch_repr, confidence_repr, approx_repr],
                dim=-1,
            )
        )
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat(
                    [
                        graph_repr,
                        formula_repr,
                        comp_repr,
                        struct_repr,
                        mismatch_repr,
                        confidence_repr,
                        approx_repr,
                        z_condition,
                    ],
                    dim=-1,
                )
            )
        )
        fused = gate * graph_repr + (1.0 - gate) * material_repr
        final = torch.cat(
            [
                fused,
                graph_repr,
                formula_repr,
                comp_repr,
                struct_repr,
                mismatch_repr,
                confidence_repr,
                approx_repr,
                material_repr,
                z_condition,
            ],
            dim=-1,
        )
        return self.head(final)
