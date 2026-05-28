"""Small scatter helpers implemented with pure PyTorch index_add."""

from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of src into dim_size buckets using index_add."""
    out = src.new_zeros((dim_size, src.shape[-1]))
    if src.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean rows of src into dim_size buckets."""
    out = scatter_sum(src, index, dim_size)
    counts = src.new_zeros((dim_size, 1))
    if src.numel() > 0:
        counts.index_add_(0, index, torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / counts.clamp_min(1.0)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Max rows of src into dim_size buckets."""
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    pooled = []
    for bucket in range(dim_size):
        values = src[index == bucket]
        if values.numel() == 0:
            pooled.append(src.new_zeros((src.shape[-1],)))
        else:
            pooled.append(values.max(dim=0).values)
    return torch.stack(pooled, dim=0)

