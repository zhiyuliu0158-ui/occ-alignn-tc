"""Regression metrics and prediction table helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def regression_metrics(y_true_tc: np.ndarray, y_pred_tc: np.ndarray) -> dict[str, float]:
    """Compute raw-Tc and log1p-Tc regression metrics."""
    y_true_tc = np.asarray(y_true_tc, dtype=float)
    y_pred_tc = np.asarray(y_pred_tc, dtype=float)
    mask = np.isfinite(y_true_tc) & np.isfinite(y_pred_tc)
    if mask.sum() == 0:
        return {
            "MAE_K": float("nan"),
            "RMSE_K": float("nan"),
            "R2_raw_Tc": float("nan"),
            "MSLE": float("nan"),
            "MAE_log1p": float("nan"),
            "RMSE_log1p": float("nan"),
        }
    true = np.clip(y_true_tc[mask], 0.0, None)
    pred = np.clip(y_pred_tc[mask], 0.0, None)
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom > 0 and len(true) > 1 else float("nan")
    true_log = np.log1p(true)
    pred_log = np.log1p(pred)
    log_error = pred_log - true_log
    return {
        "MAE_K": mae,
        "RMSE_K": rmse,
        "R2_raw_Tc": r2,
        "MSLE": float(np.mean(log_error**2)),
        "MAE_log1p": float(np.mean(np.abs(log_error))),
        "RMSE_log1p": float(np.sqrt(np.mean(log_error**2))),
    }


def make_prediction_frame(
    sample_ids: list[str],
    cif_paths: list[str],
    formulas: list[str],
    metadata: list[dict[str, object]],
    true_tc: Iterable[float],
    pred_tc: Iterable[float],
    logtc_mu: Iterable[float],
    logtc_logvar: Iterable[float],
) -> pd.DataFrame:
    """Create a rich prediction DataFrame with original metadata columns."""
    rows: list[dict[str, object]] = []
    true_tc_list = list(true_tc)
    pred_tc_list = list(pred_tc)
    mu_list = list(logtc_mu)
    logvar_list = list(logtc_logvar)
    sigma = np.exp(0.5 * np.asarray(logvar_list, dtype=float))
    for idx, sample_id in enumerate(sample_ids):
        row = dict(metadata[idx]) if idx < len(metadata) else {}
        row.update(
            {
                "sample_id": sample_id,
                "cif_path": cif_paths[idx],
                "formula": formulas[idx],
                "Tc_true_K": float(true_tc_list[idx]),
                "Tc_pred_K": float(pred_tc_list[idx]),
                "logtc_mu": float(mu_list[idx]),
                "logtc_sigma": float(sigma[idx]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Compute metrics for available metadata groups."""
    rows: list[dict[str, object]] = []
    for column in group_columns:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False):
            metrics = regression_metrics(
                group["Tc_true_K"].to_numpy(),
                group["Tc_pred_K"].to_numpy(),
            )
            metrics.update({"group_column": column, "group_value": value, "n": len(group)})
            rows.append(metrics)
    return pd.DataFrame(rows)
