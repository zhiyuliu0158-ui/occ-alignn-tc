"""Regression losses for log(1+Tc)."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log likelihood without constant term."""
    return (0.5 * torch.exp(-logvar) * (target - mu) ** 2 + 0.5 * logvar).mean()


def regression_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Select and compute the configured regression loss."""
    if loss_name == "gaussian_nll":
        return gaussian_nll_loss(outputs["logtc_mu"], outputs["logtc_logvar"], target)
    if loss_name == "huber":
        return F.huber_loss(outputs["logtc_mu"], target)
    raise ValueError(f"Unsupported regression_loss: {loss_name}")
