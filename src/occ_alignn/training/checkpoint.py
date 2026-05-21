"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
) -> None:
    """Save a model checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer states."""
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
