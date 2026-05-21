"""Gaussian radial basis function expansions."""

from __future__ import annotations

import torch
from torch import nn


class GaussianRBF(nn.Module):
    """Gaussian RBF expansion with fixed centers."""

    def __init__(self, start: float, stop: float, n_centers: int) -> None:
        super().__init__()
        centers = torch.linspace(start, stop, n_centers)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0]) if n_centers > 1 else 1.0
        self.gamma = 1.0 / max(spacing, 1e-6) ** 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Expand values into Gaussian RBF features."""
        return torch.exp(-self.gamma * (values[..., None] - self.centers) ** 2)


def gaussian_rbf_np(values, start: float, stop: float, n_centers: int):
    """NumPy equivalent used during graph construction."""
    import numpy as np

    centers = np.linspace(start, stop, n_centers, dtype=np.float32)
    spacing = float(centers[1] - centers[0]) if n_centers > 1 else 1.0
    gamma = 1.0 / max(spacing, 1e-6) ** 2
    return np.exp(-gamma * (np.asarray(values, dtype=np.float32)[..., None] - centers) ** 2)

