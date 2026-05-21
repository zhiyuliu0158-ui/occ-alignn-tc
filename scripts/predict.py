"""Predict Tc for CIF candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.dataset import CifTcDataset
from occ_alignn.models.occ_alignn_full_tc import PBOccALIGNNFullTc
from occ_alignn.training.checkpoint import load_checkpoint
from occ_alignn.training.evaluator import predict_loader
from occ_alignn.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_csv", default=None)
    parser.add_argument("--cif_dir", default=None)
    parser.add_argument("--conditions_csv", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _folder_dataframe(cif_dir: str | Path, conditions_csv: str | Path | None) -> pd.DataFrame:
    rows = []
    for cif_path in sorted(Path(cif_dir).glob("*.cif")):
        rows.append(
            {
                "sample_id": cif_path.stem,
                "cif_path": str(cif_path),
                "pressure_GPa": "unknown",
                "magnetic_field_T": "unknown",
                "field_direction": "unknown",
                "structure_source": "unknown",
                "match_type": "unknown",
                "fidelity": "unknown",
            }
        )
    df = pd.DataFrame(rows)
    if conditions_csv is not None and Path(conditions_csv).exists():
        cond = pd.read_csv(conditions_csv)
        if "sample_id" in cond.columns:
            df = df.drop(columns=[c for c in cond.columns if c in df.columns and c != "sample_id"]).merge(
                cond,
                on="sample_id",
                how="left",
            )
        elif "cif_path" in cond.columns:
            df = df.drop(columns=[c for c in cond.columns if c in df.columns and c != "cif_path"]).merge(
                cond,
                on="cif_path",
                how="left",
            )
    return df


def main() -> None:
    args = parse_args()
    if args.data_csv is None and args.cif_dir is None:
        raise ValueError("Provide either --data_csv or --cif_dir.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    if args.data_csv:
        data_csv = Path(args.data_csv)
        df = pd.read_csv(data_csv)
        csv_dir = data_csv.parent
    else:
        df = _folder_dataframe(args.cif_dir, args.conditions_csv)
        csv_dir = Path(".")
    if "Tc_K" not in df.columns:
        df["Tc_K"] = float("nan")
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"candidate_{idx}" for idx in range(len(df))]
    dataset = CifTcDataset(
        df,
        config,
        csv_dir=csv_dir,
        split=None,
        split_column=str(data_cfg.get("split_column", "split")),
        require_target=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", data_cfg.get("num_workers", 0))),
        collate_fn=collate_graphs,
    )
    model = PBOccALIGNNFullTc(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    frame = predict_loader(model, loader, device)
    out = ensure_dir(args.output_dir)
    requested = [
        "sample_id",
        "formula",
        "Tc_pred_K",
        "logtc_mu",
        "logtc_sigma",
        "pressure_GPa",
        "pressure_is_unknown",
        "magnetic_field_T",
        "field_is_unknown",
        "field_direction",
        "structure_source",
        "match_type",
        "fidelity",
        "cif_path",
    ]
    columns = [column for column in requested if column in frame.columns]
    frame[columns].to_csv(out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
