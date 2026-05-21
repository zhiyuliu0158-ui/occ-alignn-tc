"""Evaluate a trained P-B-Occ-ALIGNN-full-Tc checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.dataset import CifTcDataset, load_dataframe
from occ_alignn.data.split import ensure_split
from occ_alignn.models.occ_alignn_full_tc import PBOccALIGNNFullTc
from occ_alignn.training.checkpoint import load_checkpoint
from occ_alignn.training.evaluator import predict_loader, save_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    data_csv = Path(args.data_csv)
    df = load_dataframe(data_csv)
    split_column = str(data_cfg.get("split_column", "split"))
    df = ensure_split(
        df,
        split_column=split_column,
        group_column=str(data_cfg.get("group_column", "parent_cif_id")),
        train_frac=float(data_cfg.get("train_frac", 0.8)),
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        test_frac=float(data_cfg.get("test_frac", 0.1)),
        seed=int(training_cfg.get("seed", 42)),
    )
    dataset = CifTcDataset(df, config, data_csv.parent, split=args.split, split_column=split_column)
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
    save_evaluation_outputs(frame, args.output_dir, prefix=args.split)


if __name__ == "__main__":
    main()
