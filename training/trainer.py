"""Training loop with checkpoint save / resume, early stopping, AMP."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from utils.metrics import compute_metrics


def build_optimizer(model, cfg):
    name = cfg.get("name", "adam").lower()
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 0.0)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg):
    name = cfg.get("name", "none").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.get("t_max", 50),
                                 eta_min=cfg.get("eta_min", 1e-5))
    if name == "step":
        return StepLR(optimizer, step_size=cfg.get("step_size", 20),
                      gamma=cfg.get("gamma", 0.5))
    return None


def build_loss(cfg):
    name = cfg.get("name", "cross_entropy").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    raise ValueError(f"Unknown loss: {name}")


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, loss_fn, device,
                 cfg, logger, tb_writer=None, checkpoint_dir=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.tb = tb_writer
        self.ckpt_dir = Path(checkpoint_dir or "checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = int(cfg["epochs"])
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.use_amp = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
        }
        self.start_epoch = 0
        self.best_metric = -float("inf")
        self.epochs_without_improve = 0
        self.es_cfg = cfg.get("early_stopping", {})

    # ------------------------------------------------------------------
    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        all_pred, all_true, all_prob = [], [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        desc = "train" if train else "val"
        iterator = tqdm(loader, desc=desc, leave=False, ncols=90)

        with ctx:
            for x, y in iterator:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += float(loss.item()) * x.size(0)
                probs = torch.softmax(logits.detach().float(), dim=-1)
                all_pred.append(probs.argmax(dim=-1).cpu().numpy())
                all_true.append(y.cpu().numpy())
                all_prob.append(probs.cpu().numpy())
                iterator.set_postfix(loss=f"{loss.item():.3f}")

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_prob = np.concatenate(all_prob)
        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = avg_loss
        return metrics

    # ------------------------------------------------------------------
    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_m = self._run_epoch(self.train_loader, train=True)
            val_m = self._run_epoch(self.val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_acc"].append(train_m["accuracy"])
            self.history["val_acc"].append(val_m["accuracy"])
            self.history["train_f1"].append(train_m["macro_f1"])
            self.history["val_f1"].append(val_m["macro_f1"])

            self.logger.info(
                f"Epoch {epoch + 1:03d}/{self.epochs} | "
                f"train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} f1={train_m['macro_f1']:.4f} | "
                f"val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} f1={val_m['macro_f1']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

            if self.tb is not None:
                self.tb.add_scalar("loss/train", train_m["loss"], epoch + 1)
                self.tb.add_scalar("loss/val",   val_m["loss"],   epoch + 1)
                self.tb.add_scalar("acc/train",  train_m["accuracy"], epoch + 1)
                self.tb.add_scalar("acc/val",    val_m["accuracy"],   epoch + 1)
                self.tb.add_scalar("f1/train",   train_m["macro_f1"], epoch + 1)
                self.tb.add_scalar("f1/val",     val_m["macro_f1"],   epoch + 1)

            self._save_last(epoch)
            self._maybe_save_best(val_m, epoch)

            if self._should_stop():
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history

    # ------------------------------------------------------------------
    def _save_last(self, epoch):
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
        }
        torch.save(state, self.ckpt_dir / "last.pth")

    def _maybe_save_best(self, val_m, epoch):
        monitor = self.es_cfg.get("monitor", "val_macro_f1")
        mode = self.es_cfg.get("mode", "max")
        key = monitor.split("_", 1)[1] if monitor.startswith("val_") else monitor
        if key not in val_m:
            return

        score = val_m[key]
        better = (score > self.best_metric) if mode == "max" else (score < self.best_metric)
        if better or self.best_metric == -float("inf"):
            self.best_metric = score
            self.epochs_without_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "history": self.history,
                "monitor": monitor,
            }, self.ckpt_dir / "best.pth")
            self.logger.info(f"  ↳ new best {monitor}={score:.4f} → saved best.pth")
        else:
            self.epochs_without_improve += 1

    def _should_stop(self):
        if not self.es_cfg.get("enabled", False):
            return False
        return self.epochs_without_improve >= int(self.es_cfg.get("patience", 10))

    # ------------------------------------------------------------------
    def resume(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            self.logger.warning(f"Resume checkpoint not found: {ckpt_path}")
            return
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler is not None and state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])
        self.start_epoch = state.get("epoch", 0)
        self.best_metric = state.get("best_metric", -float("inf"))
        self.history = state.get("history", self.history)
        self.logger.info(f"Resumed from {ckpt_path} at epoch {self.start_epoch}")