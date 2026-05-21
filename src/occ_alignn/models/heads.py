"""Regression heads for Tc prediction."""

from __future__ import annotations

import torch
from torch import nn

from occ_alignn.nn.film import mlp


class GaussianTcHead(nn.Module):
    """Predict log(1+Tc) mean and log variance."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mu = mlp(input_dim, hidden_dim, 1, dropout)
        self.logvar = mlp(input_dim, hidden_dim, 1, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return log-space Gaussian parameters and raw Tc prediction."""
        logtc_mu = self.mu(x).squeeze(-1)
        logtc_logvar = self.logvar(x).squeeze(-1).clamp(-6.0, 6.0)
        tc_pred_k = torch.expm1(logtc_mu).clamp_min(0.0)
        return {
            "logtc_mu": logtc_mu,
            "logtc_logvar": logtc_logvar,
            "Tc_pred_K": tc_pred_k,
        }
