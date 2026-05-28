"""Evaluation loops and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from occ_alignn.data.collate import GraphBatch
from occ_alignn.training.metrics import grouped_metrics, make_prediction_frame, regression_metrics
from occ_alignn.utils.io import ensure_dir, write_json


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> pd.DataFrame:
    """Run model inference over a DataLoader and return predictions."""
    model.eval()
    all_sample_ids: list[str] = []
    all_cif_paths: list[str] = []
    all_formulas: list[str] = []
    all_metadata: list[dict[str, object]] = []
    true_tc: list[float] = []
    pred_tc: list[float] = []
    mu_values: list[float] = []
    logvar_values: list[float] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        assert isinstance(batch, GraphBatch)
        batch = batch.to(device)
        outputs = model(batch)
        all_sample_ids.extend(batch.sample_ids)
        all_cif_paths.extend(batch.cif_paths)
        all_formulas.extend(batch.formulas)
        all_metadata.extend(batch.metadata)
        true_tc.extend(batch.tc_k.detach().cpu().numpy().tolist())
        pred_tc.extend(outputs["Tc_pred_K"].detach().cpu().numpy().tolist())
        mu_values.extend(outputs["logtc_mu"].detach().cpu().numpy().tolist())
        logvar_values.extend(outputs["logtc_logvar"].detach().cpu().numpy().tolist())
    return make_prediction_frame(
        all_sample_ids,
        all_cif_paths,
        all_formulas,
        all_metadata,
        true_tc,
        pred_tc,
        mu_values,
        logvar_values,
    )


def _save_scatter(x: np.ndarray, y: np.ndarray, path: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=18, alpha=0.75)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        low = float(min(np.min(x[finite]), np.min(y[finite])))
        high = float(max(np.max(x[finite]), np.max(y[finite])))
        plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_evaluation_outputs(frame: pd.DataFrame, output_dir: str | Path, prefix: str = "test") -> dict[str, float]:
    """Save metrics, grouped metrics, and diagnostic plots."""
    out = ensure_dir(output_dir)
    frame.to_csv(out / f"predictions_{prefix}.csv", index=False)
    if frame.empty:
        raise ValueError(f"Cannot evaluate empty prediction frame for split {prefix!r}.")
    required_columns = {"Tc_true_K", "Tc_pred_K"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot evaluate predictions without columns: {missing}.")
    metrics = regression_metrics(frame["Tc_true_K"].to_numpy(), frame["Tc_pred_K"].to_numpy())
    write_json(metrics, out / f"metrics_{prefix}.json")
    diagnostic_frame = frame.copy()
    diagnostic_frame["tc_bin"] = pd.cut(
        diagnostic_frame["Tc_true_K"],
        [-0.1, 1, 5, 10, 20, 40, 80, 120, float("inf")],
        labels=["<=1", "1-5", "5-10", "10-20", "20-40", "40-80", "80-120", ">120"],
    )
    pressure = pd.to_numeric(diagnostic_frame.get("pressure_GPa", 0.0), errors="coerce").fillna(0.0)
    field = pd.to_numeric(diagnostic_frame.get("magnetic_field_T", 0.0), errors="coerce").fillna(0.0)
    diagnostic_frame["pressure_bin"] = pd.cut(
        pressure,
        [-0.1, 0, 1, 10, 50, 100, float("inf")],
        labels=["0", "0-1", "1-10", "10-50", "50-100", ">100"],
    )
    diagnostic_frame["field_bin"] = pd.cut(
        field,
        [-0.1, 0, 0.01, 1, 10, float("inf")],
        labels=["0", "0-0.01", "0.01-1", "1-10", ">10"],
    )
    group_cols = [
        "match_type",
        "fidelity",
        "structure_source",
        "family",
        "chemical_system",
        "pressure_is_unknown",
        "field_is_unknown",
        "tc_bin",
        "pressure_bin",
        "field_bin",
    ]
    groups = grouped_metrics(diagnostic_frame, group_cols)
    if not groups.empty:
        groups.to_csv(out / f"grouped_metrics_{prefix}.csv", index=False)
    formula_standardized = (
        diagnostic_frame["formula_standardized"].astype(str)
        if "formula_standardized" in diagnostic_frame.columns
        else pd.Series("", index=diagnostic_frame.index)
    )
    match_type = (
        diagnostic_frame["match_type"].astype(str)
        if "match_type" in diagnostic_frame.columns
        else pd.Series("", index=diagnostic_frame.index)
    )
    subset_rows: list[dict[str, object]] = []
    subset_masks = {
        "all": pd.Series(True, index=diagnostic_frame.index),
        "exclude_H3S": formula_standardized != "H3S",
        "exclude_Tc_gt_120": diagnostic_frame["Tc_true_K"] <= 120.0,
        "high_pressure_gt_50": pressure > 50.0,
        "formula_exact": match_type == "formula_exact",
        "formula_similarity": match_type == "formula_similarity",
    }
    for name, mask in subset_masks.items():
        subset = diagnostic_frame[mask.fillna(False)]
        row = regression_metrics(subset["Tc_true_K"].to_numpy(), subset["Tc_pred_K"].to_numpy())
        row.update({"subset": name, "n": int(len(subset))})
        subset_rows.append(row)
    pd.DataFrame(subset_rows).to_csv(out / f"subset_metrics_{prefix}.csv", index=False)
    worst = diagnostic_frame.assign(
        abs_error_K=(diagnostic_frame["Tc_pred_K"] - diagnostic_frame["Tc_true_K"]).abs()
    ).sort_values("abs_error_K", ascending=False)
    worst.to_csv(out / f"worst_errors_{prefix}.csv", index=False)

    true_tc = frame["Tc_true_K"].to_numpy(dtype=float)
    pred_tc = frame["Tc_pred_K"].to_numpy(dtype=float)
    _save_scatter(true_tc, pred_tc, out / f"pred_vs_true_{prefix}.png", "True Tc (K)", "Predicted Tc (K)")
    _save_scatter(
        np.log1p(np.clip(true_tc, 0, None)),
        np.log1p(np.clip(pred_tc, 0, None)),
        out / f"log_pred_vs_log_true_{prefix}.png",
        "True log(1+Tc)",
        "Predicted log(1+Tc)",
    )
    plt.figure(figsize=(6, 4))
    plt.hist(pred_tc - true_tc, bins=30, alpha=0.8)
    plt.xlabel("Prediction residual (K)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / f"residual_hist_{prefix}.png", dpi=180)
    plt.close()

    if "logtc_sigma" in frame.columns:
        plt.figure(figsize=(5, 4))
        plt.scatter(frame["logtc_sigma"], np.abs(pred_tc - true_tc), s=18, alpha=0.75)
        plt.xlabel("Predicted log-space sigma")
        plt.ylabel("Absolute error (K)")
        plt.tight_layout()
        plt.savefig(out / f"uncertainty_vs_abs_error_{prefix}.png", dpi=180)
        plt.close()

    for column in ("family", "match_type"):
        if column in frame.columns:
            grouped = frame.groupby(column).apply(
                lambda g: regression_metrics(g["Tc_true_K"].to_numpy(), g["Tc_pred_K"].to_numpy())["MAE_K"]
            )
            if len(grouped) > 0:
                plt.figure(figsize=(max(6, 0.35 * len(grouped)), 4))
                grouped.sort_values().plot(kind="bar")
                plt.ylabel("MAE_K")
                plt.tight_layout()
                plt.savefig(out / f"metrics_by_{column}_{prefix}.png", dpi=180)
                plt.close()
    return metrics
