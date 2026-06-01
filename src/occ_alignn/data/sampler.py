"""Sampling helpers for imbalanced Tc regimes."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from occ_alignn.analysis.regimes import add_regime_columns


def regime_sample_weights(frame: pd.DataFrame, training_cfg: dict[str, Any]) -> torch.DoubleTensor:
    """Compute per-row sampler weights from standard high-risk regimes."""
    enriched = add_regime_columns(frame)
    weights = pd.Series(1.0, index=enriched.index, dtype=float)
    weights.loc[enriched["is_high_tc"]] *= float(training_cfg.get("sampler_high_tc_weight", 1.5))
    weights.loc[enriched["is_very_high_tc"]] *= float(training_cfg.get("sampler_very_high_tc_weight", 2.0))
    weights.loc[enriched["is_high_pressure"]] *= float(training_cfg.get("sampler_high_pressure_weight", 1.5))
    weights.loc[enriched["is_cuprate"]] *= float(training_cfg.get("sampler_cuprate_weight", 1.5))
    weights.loc[enriched["is_synthetic_doped"]] *= float(training_cfg.get("sampler_synthetic_doped_weight", 1.25))
    weights.loc[enriched["is_hydride_high_pressure"]] *= float(
        training_cfg.get("sampler_hydride_high_pressure_weight", 1.5)
    )
    max_weight = float(training_cfg.get("sampler_max_weight", 4.0))
    weights = weights.clip(lower=0.0, upper=max_weight)
    return torch.as_tensor(weights.to_numpy(dtype=float), dtype=torch.double)


def build_balanced_sampler(dataset: object, training_cfg: dict[str, Any]) -> WeightedRandomSampler | None:
    """Build an optional weighted sampler for CifTcDataset-like objects."""
    if not bool(training_cfg.get("use_balanced_sampler", False)):
        return None
    dataframe = getattr(dataset, "dataframe", None)
    if dataframe is None or len(dataframe) == 0:
        return None
    weights = regime_sample_weights(dataframe, training_cfg)
    num_samples = int(training_cfg.get("balanced_sampler_num_samples", len(dataframe)))
    replacement = bool(training_cfg.get("balanced_sampler_replacement", True))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=replacement)
