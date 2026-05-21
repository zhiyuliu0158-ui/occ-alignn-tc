"""Graph pooling layers."""

from __future__ import annotations

import torch

from occ_alignn.nn.scatter import scatter_max, scatter_mean


def mean_pool(node_h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Mean-pool node embeddings per graph."""
    return scatter_mean(node_h, batch, dim_size=num_graphs)


def mean_max_pool(node_h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Concatenate mean and max pooled node embeddings."""
    return torch.cat(
        [
            scatter_mean(node_h, batch, dim_size=num_graphs),
            scatter_max(node_h, batch, dim_size=num_graphs),
        ],
        dim=-1,
    )
