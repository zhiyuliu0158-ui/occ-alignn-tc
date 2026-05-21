"""Pure PyTorch ALIGNN-style message passing."""

from __future__ import annotations

import torch
from torch import nn

from occ_alignn.nn.film import mlp
from occ_alignn.nn.scatter import scatter_sum


class ALIGNNLayer(nn.Module):
    """One ALIGNN-style layer: line-graph edge update then bond-graph node update."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.line_mlp = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.line_gate = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.bond_mlp = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.bond_gate = mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        angle_h: torch.Tensor,
        edge_index: torch.Tensor,
        line_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update node and edge embeddings while keeping angle embeddings fixed."""
        if edge_h.numel() > 0 and line_edge_index.numel() > 0 and angle_h.numel() > 0:
            line_src = line_edge_index[0]
            line_dst = line_edge_index[1]
            line_input = torch.cat(
                [edge_h[line_src], edge_h[line_dst], angle_h],
                dim=-1,
            )
            message = torch.sigmoid(self.line_gate(line_input)) * self.line_mlp(line_input)
            edge_update = scatter_sum(message, line_dst, dim_size=edge_h.shape[0])
            edge_h = self.edge_norm(edge_h + self.dropout(edge_update))

        if edge_h.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            bond_input = torch.cat([node_h[src], node_h[dst], edge_h], dim=-1)
            message = torch.sigmoid(self.bond_gate(bond_input)) * self.bond_mlp(bond_input)
            node_update = scatter_sum(message, dst, dim_size=node_h.shape[0])
            node_h = self.node_norm(node_h + self.dropout(node_update))

        return node_h, edge_h, angle_h
