"""Train P-B-Occ-ALIGNN-full-Tc."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from occ_alignn.data.collate import collate_graphs
from occ_alignn.data.dataset import CifTcDataset, load_dataframe
from occ_alignn.data.split import ensure_split
from occ_alignn.models.occ_alignn_full_tc import PBOccALIGNNFullTc
from occ_alignn.training.trainer import Trainer
from occ_alignn.utils.config import load_config
from occ_alignn.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    set_seed(int(training_cfg.get("seed", 42)))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

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
    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(training_cfg.get("num_workers", data_cfg.get("num_workers", 0)))
    train_dataset = CifTcDataset(df, config, data_csv.parent, split="train", split_column=split_column)
    val_dataset = CifTcDataset(df, config, data_csv.parent, split="val", split_column=split_column)
    test_dataset = CifTcDataset(df, config, data_csv.parent, split="test", split_column=split_column)
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        val_dataset = test_dataset if len(test_dataset) > 0 else train_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_graphs,
        )
        if len(test_dataset) > 0
        else None
    )
    model = PBOccALIGNNFullTc(config)
    trainer = Trainer(model, config, args.output_dir, device=device)
    trainer.fit(train_loader, val_loader, test_loader, resume_path=args.resume)


if __name__ == "__main__":
    main()
