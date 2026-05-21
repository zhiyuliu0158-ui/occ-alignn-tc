"""Training loop for P-B-Occ-ALIGNN-full-Tc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from occ_alignn.data.collate import GraphBatch
from occ_alignn.training.checkpoint import load_checkpoint, save_checkpoint
from occ_alignn.training.evaluator import predict_loader, save_evaluation_outputs
from occ_alignn.training.losses import regression_loss
from occ_alignn.utils.config import save_config
from occ_alignn.utils.io import ensure_dir, write_json
from occ_alignn.utils.logging import get_logger

LOGGER = get_logger(__name__)


class Trainer:
    """Manage optimization, validation, early stopping, and artifacts."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        output_dir: str | Path,
        device: torch.device | str,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        self.device = torch.device(device)
        training_cfg = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        )
        scheduler_name = str(training_cfg.get("scheduler", "reduce_on_plateau"))
        if scheduler_name == "cosine":
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, int(training_cfg.get("epochs", 200))),
                )
            )
        elif scheduler_name == "none":
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        self.loss_name = str(training_cfg.get("regression_loss", "gaussian_nll"))
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.start_epoch = 0
        self.best_metric = float("inf")

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume model and optimizer state."""
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            map_location=self.device,
        )
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_metric = float(checkpoint.get("best_metric", float("inf")))

    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
        for batch in progress:
            assert isinstance(batch, GraphBatch)
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch)
            loss = regression_loss(outputs, batch.target, self.loss_name)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        return float(sum(losses) / max(1, len(losses)))

    @torch.no_grad()
    def _run_eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch in tqdm(loader, desc="validation", leave=False):
            assert isinstance(batch, GraphBatch)
            batch = batch.to(self.device)
            outputs = self.model(batch)
            losses.append(float(regression_loss(outputs, batch.target, self.loss_name).detach().cpu()))
        return float(sum(losses) / max(1, len(losses)))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        resume_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Train the model and write all training artifacts."""
        if resume_path is not None:
            self.resume(resume_path)
        save_config(self.config, self.output_dir / "config.yaml")
        training_cfg = self.config.get("training", {})
        epochs = int(training_cfg.get("epochs", 200))
        patience = int(training_cfg.get("early_stopping_patience", 30))
        history: list[dict[str, float]] = []
        stale_epochs = 0
        for epoch in range(self.start_epoch, epochs):
            train_loss = self._run_train_epoch(train_loader, epoch)
            val_loss = self._run_eval_loss(val_loader)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            LOGGER.info("epoch=%d train_loss=%.5f val_loss=%.5f", epoch, train_loss, val_loss)
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                self.config,
                epoch,
                self.best_metric,
            )
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                stale_epochs = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.config,
                    epoch,
                    self.best_metric,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    LOGGER.info("early stopping at epoch %d", epoch)
                    break

        write_json({"history": history, "best_val_loss": self.best_metric}, self.output_dir / "metrics.json")
        load_checkpoint(self.output_dir / "best.pt", model=self.model, map_location=self.device)
        val_frame = predict_loader(self.model, val_loader, self.device)
        val_metrics = save_evaluation_outputs(val_frame, self.output_dir, prefix="val")
        if test_loader is not None:
            test_frame = predict_loader(self.model, test_loader, self.device)
            save_evaluation_outputs(test_frame, self.output_dir, prefix="test")
        return val_metrics
