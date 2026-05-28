"""Regression losses for log(1+Tc)."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return loss.mean()
    weight = weight.to(dtype=loss.dtype, device=loss.device)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def gaussian_nll_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian negative log likelihood without constant term."""
    loss = 0.5 * torch.exp(-logvar) * (target - mu) ** 2 + 0.5 * logvar
    return _weighted_mean(loss, weight)


def _mixed_log_raw_huber_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    raw_weight: float = 0.15,
    raw_scale: float = 50.0,
) -> torch.Tensor:
    """Huber loss on log(1+Tc) plus a small raw-Tc term.

    The model is still trained mainly in log space, but the raw-Tc term keeps
    high-temperature outliers from becoming nearly invisible after log scaling.
    """
    log_loss = F.huber_loss(mu, target, reduction="none")
    pred_tc = torch.expm1(mu).clamp_min(0.0)
    true_tc = torch.expm1(target).clamp_min(0.0)
    raw_loss = F.huber_loss(pred_tc / raw_scale, true_tc / raw_scale, reduction="none")
    return _weighted_mean(log_loss + raw_weight * raw_loss, weight)


def regression_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_name: str,
    weight: torch.Tensor | None = None,
    raw_weight: float = 0.15,
    raw_scale: float = 50.0,
) -> torch.Tensor:
    """Select and compute the configured regression loss."""
    if loss_name == "gaussian_nll":
        return gaussian_nll_loss(outputs["logtc_mu"], outputs["logtc_logvar"], target, weight=weight)
    if loss_name == "huber":
        loss = F.huber_loss(outputs["logtc_mu"], target, reduction="none")
        return _weighted_mean(loss, weight)
    if loss_name == "mixed_log_raw_huber":
        return _mixed_log_raw_huber_loss(
            outputs["logtc_mu"],
            target,
            weight=weight,
            raw_weight=raw_weight,
            raw_scale=raw_scale,
        )
    raise ValueError(f"Unsupported regression_loss: {loss_name}")
