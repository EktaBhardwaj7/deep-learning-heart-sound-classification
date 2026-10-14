"""
Multi-task trainer with alternating batches, loss masking, class weights,
and optional backbone freezing for the first N epochs.
"""
from __future__ import annotations

from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import compute_metrics


class MultiTaskTrainer:
    def __init__(self, model, loaders, optimizer, scheduler,
                 device, cfg, logger, tb_writer=None, checkpoint_dir=None,
                 yaseen_class_weights=None, circor_class_weights=None):
        self.model = model.to(device)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.tb = tb_writer
        self.ckpt_dir = Path(checkpoint_dir or "checkpoints/mix")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = int(cfg["epochs"])
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.lambda_circor = float(cfg.get("lambda_circor", 1.0))
        self.freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0))
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        smoothing = cfg.get("label_smoothing", 0.05)

        w_y = None
        if yaseen_class_weights is not None:
            w_y = torch.tensor(yaseen_class_weights, dtype=torch.float32, device=device)
        self.loss_y = nn.CrossEntropyLoss(weight=w_y, label_smoothing=smoothing)

        w_c = None
        if circor_class_weights is not None:
            w_c = torch.tensor(circor_class_weights, dtype=torch.float32, device=device)
            self.logger.info(f"CirCor class weights: {w_c.tolist()}")
        self.loss_c = nn.CrossEntropyLoss(weight=w_c, label_smoothing=smoothing)

        self.history = {k: [] for k in
                        ["train_loss", "val_loss",
                         "y_val_acc", "y_val_f1", "c_val_acc", "c_val_f1"]}
        self.start_epoch = 0
        self.best = -float("inf")
        self.waited = 0
        self.es_cfg = cfg.get("early_stopping", {})
        self._backbone_frozen = None

    # ------------------------------------------------------------------
    def _set_backbone_frozen(self, frozen: bool):
        if self._backbone_frozen == frozen:
            return
        for p in self.model.backbone.parameters():
            p.requires_grad = not frozen
        self._backbone_frozen = frozen
        state = "FROZEN" if frozen else "UNFROZEN"
        self.logger.info(f"  backbone is now {state}")

    # ------------------------------------------------------------------
    def _step(self, x, y, task, train):
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(x, task=task)
            loss = self.loss_y(logits, y) if task == "yaseen" \
                   else self.lambda_circor * self.loss_c(logits, y)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
        return loss.item(), preds, y.cpu().numpy(), probs.cpu().numpy()

    # ------------------------------------------------------------------
    def _epoch(self, train: bool):
        load_y = self.loaders["yaseen_train" if train else "yaseen_val"]
        load_c = self.loaders["circor_train" if train else "circor_val"]

        n_steps = max(len(load_y), len(load_c))
        it_y = cycle(load_y)
        it_c = cycle(load_c)

        # Balanced per-sample exposure: Yaseen once every K steps, CirCor on
        # the rest, where K = len(circor) // len(yaseen).
        if len(load_y) > 0 and len(load_c) > 0:
            steps_per_yaseen = max(1, len(load_c) // len(load_y))
        else:
            steps_per_yaseen = 2

        if train:
            self.logger.info(f"  balanced exposure: Yaseen every {steps_per_yaseen} steps (n_steps={n_steps})")

        total_loss = 0.0
        n = 0
        y_true, y_pred, y_prob = [], [], []
        c_true, c_pred, c_prob = [], [], []

        iterator = tqdm(range(n_steps), desc="train" if train else "val",
                        leave=False, ncols=90)
        for step in iterator:
            use_yaseen = (len(load_c) == 0) or (step % steps_per_yaseen == 0)
            if use_yaseen and len(load_y) > 0:
                x, y = next(it_y)
                loss, preds, gts, probs = self._step(x, y, "yaseen", train)
                y_true.append(gts); y_pred.append(preds); y_prob.append(probs)
            else:
                x, y = next(it_c)
                loss, preds, gts, probs = self._step(x, y, "circor", train)
                c_true.append(gts); c_pred.append(preds); c_prob.append(probs)

            total_loss += loss
            n += 1
            iterator.set_postfix(loss=f"{loss:.3f}")

        avg_loss = total_loss / max(n, 1)
        ym = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred),
                             np.concatenate(y_prob)) if y_true else {}
        cm = compute_metrics(np.concatenate(c_true), np.concatenate(c_pred),
                             np.concatenate(c_prob)) if c_true else {}
        return avg_loss, ym, cm

    # ------------------------------------------------------------------
    def fit(self):
        for ep in range(self.start_epoch, self.epochs):
            # Backbone freeze schedule
            if ep < self.freeze_backbone_epochs:
                self._set_backbone_frozen(True)
            else:
                self._set_backbone_frozen(False)

            tr_loss, _, _ = self._epoch(train=True)
            va_loss, va_y, va_c = self._epoch(train=False)
            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["y_val_acc"].append(va_y.get("accuracy", 0.0))
            self.history["y_val_f1"].append(va_y.get("macro_f1", 0.0))
            self.history["c_val_acc"].append(va_c.get("accuracy", 0.0))
            self.history["c_val_f1"].append(va_c.get("macro_f1", 0.0))

            self.logger.info(
                f"Epoch {ep+1:03d}/{self.epochs} | "
                f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
                f"Yaseen: acc={va_y.get('accuracy',0):.4f} f1={va_y.get('macro_f1',0):.4f} | "
                f"CirCor: acc={va_c.get('accuracy',0):.4f} f1={va_c.get('macro_f1',0):.4f}"
            )

            if self.tb is not None:
                self.tb.add_scalar("mix/loss_train", tr_loss, ep + 1)
                self.tb.add_scalar("mix/loss_val", va_loss, ep + 1)
                self.tb.add_scalar("mix/y_f1", self.history["y_val_f1"][-1], ep + 1)
                self.tb.add_scalar("mix/c_f1", self.history["c_val_f1"][-1], ep + 1)

            self._save(ep, "last.pth")

            y_f1 = self.history["y_val_f1"][-1]
            c_f1 = self.history["c_val_f1"][-1]
            score = 0.4 * y_f1 + 0.6 * c_f1

            if score > self.best:
                self.best = score
                self.waited = 0
                self._save(ep, "best.pth")
                self.logger.info(f"  ↳ new best combined val F1 = {score:.4f}")
            else:
                self.waited += 1
                if self.es_cfg.get("enabled", True) and \
                   self.waited >= int(self.es_cfg.get("patience", 12)):
                    self.logger.info(f"Early stopping at epoch {ep+1}")
                    break
        return self.history

    # ------------------------------------------------------------------
    def _save(self, ep, name):
        torch.save({
            "epoch": ep + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best": self.best,
            "history": self.history,
        }, self.ckpt_dir / name)

    def resume(self, path):
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"resume not found: {path}")
            return
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model_state"])
        self.optimizer.load_state_dict(s["optimizer_state"])
        if self.scheduler and s.get("scheduler_state"):
            self.scheduler.load_state_dict(s["scheduler_state"])
        self.start_epoch = s.get("epoch", 0)
        self.best = s.get("best", -float("inf"))
        self.history = s.get("history", self.history)
        self.logger.info(f"Resumed from {path} at epoch {self.start_epoch}")
