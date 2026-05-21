"""Condition encoders and FiLM modulation."""

from __future__ import annotations

import torch
from torch import nn

from occ_alignn.featurizers.occupancy import (
    FIDELITIES,
    FIELD_DIRECTIONS,
    MATCH_TYPES,
    STRUCTURE_SOURCES,
)


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Small GELU MLP used across the model."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class ConditionEncoder(nn.Module):
    """Encode pressure, magnetic field, direction, and metadata."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        cat_dim = max(8, hidden_dim // 4)
        self.field_direction_emb = nn.Embedding(len(FIELD_DIRECTIONS), cat_dim)
        self.structure_source_emb = nn.Embedding(len(STRUCTURE_SOURCES), cat_dim)
        self.match_type_emb = nn.Embedding(len(MATCH_TYPES), cat_dim)
        self.fidelity_emb = nn.Embedding(len(FIDELITIES), cat_dim)
        self.pressure_mlp = mlp(3, hidden_dim, hidden_dim, dropout)
        self.field_mlp = mlp(3 + cat_dim, hidden_dim, hidden_dim, dropout)
        self.metadata_mlp = mlp(3 * cat_dim, hidden_dim, hidden_dim, dropout)
        self.condition_mlp = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(
        self,
        pressure: torch.Tensor,
        field: torch.Tensor,
        field_direction_id: torch.Tensor,
        structure_source_id: torch.Tensor,
        match_type_id: torch.Tensor,
        fidelity_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return one condition embedding per graph."""
        z_pressure = self.pressure_mlp(pressure)
        direction = self.field_direction_emb(field_direction_id)
        z_field = self.field_mlp(torch.cat([field, direction], dim=-1))
        metadata = torch.cat(
            [
                self.structure_source_emb(structure_source_id),
                self.match_type_emb(match_type_id),
                self.fidelity_emb(fidelity_id),
            ],
            dim=-1,
        )
        z_metadata = self.metadata_mlp(metadata)
        return self.condition_mlp(torch.cat([z_pressure, z_field, z_metadata], dim=-1))


class FiLM(nn.Module):
    """Feature-wise affine modulation from graph-level conditions."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, h: torch.Tensor, z_condition: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to rows in h using graph_index to select condition rows."""
        if h.numel() == 0:
            return h
        gamma_beta = self.to_gamma_beta(z_condition[graph_index])
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta
