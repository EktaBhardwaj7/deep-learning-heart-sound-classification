"""Training loop for one or two tasks.

- Backbone frozen for `freeze_epochs`, then released at `backbone_lr_mult` x head learning rate.
- Balanced exposure: with two tasks, batches are interleaved so each dataset is seen about once per epoch.
- Weight EMA (evaluation uses EMA weights), warm-up then cosine learning rate.
- Early stopping on a composite validation macro-F1 score.
- Every epoch writes last.pth (atomic). best.pth is written on improvement. `resume` restores everything,
  including optimiser, scheduler, EMA, RNG state and history, so training continues instead of restarting.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from .metrics import macro_f1
from .utils import load_checkpoint, progress, save_checkpoint, write_json

NUM_CLASSES = {"yaseen": 5, "circor": 2}


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # warm-up avoids early random weights
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, s):
        self.module.load_state_dict(s["module"])
        self.updates = s["updates"]


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    o = t["optimizer"]
    lr, wd = float(o["lr"]), float(o.get("weight_decay", 0.0))
    backbone = list(model.backbone.parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": heads, "lr": lr},
              {"params": backbone, "lr": lr * float(t.get("backbone_lr_mult", 1.0))}]
    name = o["name"].lower()
    if name == "adam":
        return Adam(groups, weight_decay=wd)
    if name == "adamw":
        return AdamW(groups, weight_decay=wd)
    return SGD(groups, weight_decay=wd, momentum=0.9, nesterov=True)


def build_scheduler(opt, cfg: dict):
    t = cfg["train"]
    warm = int(t.get("warmup_epochs", 0))
    total = int(t["epochs"])
    base = float(opt.param_groups[0]["lr"])
    ratio = float(t.get("eta_min", 1e-6)) / max(base, 1e-12)

    def f(ep):
        if warm > 0 and ep < warm:
            return (ep + 1) / warm
        p = min((ep - warm) / max(1, total - warm), 1.0)
        return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(opt, f)


class Trainer:
    def __init__(self, model, loaders: Dict[str, dict], cfg: dict, device, logger, ckpt_dir: Path,
                 run_dir: Path, class_weights: Dict[str, Optional[list]]):
        self.cfg, self.device, self.log = cfg, device, logger
        self.model = model.to(device)
        self.tasks = list(cfg["data"]["tasks"])
        self.train_loaders = {t: loaders[t]["train"] for t in self.tasks}
        self.val_loaders = {t: loaders[t]["val"] for t in self.tasks}
        t = cfg["train"]
        self.epochs = int(t["epochs"])
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.freeze_epochs = int(t.get("freeze_epochs", 0))
        self.exposure = t.get("exposure", "epoch")
        self.amp = bool(t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ckpt_dir, self.run_dir = Path(ckpt_dir), Path(run_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        smooth = float(t.get("label_smoothing", 0.0))
        self.loss = {}
        for task in self.tasks:
            w = class_weights.get(task)
            wt = torch.tensor(w, dtype=torch.float32, device=device) if w else None
            self.loss[task] = nn.CrossEntropyLoss(weight=wt, label_smoothing=smooth)
        self.loss_w = {k: float(v) for k, v in t.get("loss_weight", {}).items()}

        self.opt = build_optimizer(self.model, cfg)
        self.sched = build_scheduler(self.opt, cfg)
        ema = t.get("ema", {})
        self.ema = EMA(self.model, float(ema.get("decay", 0.999))) if ema.get("enabled", True) else None

        raw = {k: float(t.get("monitor_weights", {}).get(k, 1.0)) for k in self.tasks}
        z = sum(raw.values())
        self.monitor = {k: v / z for k, v in raw.items()}
        es = t.get("early_stopping", {})
        self.es_on = bool(es.get("enabled", True))
        self.patience = int(es.get("patience", 15))
        self.min_delta = float(es.get("min_delta", 1e-4))

        self.start_epoch, self.best, self.waited = 0, -math.inf, 0
        self.history: list = []
        self._frozen = None

    # ----------------------------------------------------------- internals
    def _plan(self):
        sizes = {k: len(self.train_loaders[k]) for k in self.tasks}
        if len(self.tasks) == 1:
            return [self.tasks[0]] * sizes[self.tasks[0]]
        if self.exposure == "alternate":
            n = max(sizes.values())
            return [k for _ in range(n) for k in self.tasks]
        pos = sorted(((j + 0.5) / n, k) for k, n in sizes.items() for j in range(n))
        return [k for _, k in pos]

    def _set_freeze(self, epoch: int):
        frozen = epoch < self.freeze_epochs
        if frozen != self._frozen:
            self.model.set_backbone_trainable(not frozen)
            self._frozen = frozen
            self.log.info(f"backbone {'FROZEN' if frozen else 'UNFROZEN'} at epoch {epoch + 1}")

    def _train_epoch(self) -> dict:
        self.model.train()
        if self._frozen:
            self.model.backbone.eval()           # frozen BatchNorm statistics
        plan = self._plan()
        iters = {k: _cycle(self.train_loaders[k]) for k in self.tasks}
        acc = {k: {"loss": 0.0, "n": 0, "y": [], "p": []} for k in self.tasks}
        for task in progress(plan, desc="train", leave=False, ncols=88):
            x, y, _ = next(iters[task])
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                logits = self.model(x, task=task)
            loss = self.loss_w.get(task, 1.0) * self.loss[task](logits.float(), y)
            if not torch.isfinite(loss):
                self.log.warning("non-finite loss; batch skipped")
                continue
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)
            a = acc[task]
            a["loss"] += float(loss.item())
            a["n"] += 1
            a["y"].append(y.cpu().numpy())
            a["p"].append(logits.argmax(-1).detach().cpu().numpy())
        out = {}
        for k, a in acc.items():
            if a["n"]:
                out[k] = {"loss": a["loss"] / a["n"],
                          "macro_f1": macro_f1(np.concatenate(a["y"]), np.concatenate(a["p"]), NUM_CLASSES[k])}
            else:
                out[k] = {"loss": float("nan"), "macro_f1": float("nan")}
        return out

    @torch.no_grad()
    def _validate(self) -> dict:
        net = self.ema.module if self.ema is not None else self.model
        net.eval()
        out = {}
        for task in self.tasks:
            ys, ps, tot, n = [], [], 0.0, 0
            for x, y, _ in self.val_loaders[task]:
                x, y = x.to(self.device), y.to(self.device)
                logits = net(x, task=task).float()
                tot += float(self.loss[task](logits, y).item()) * len(y)
                n += len(y)
                ys.append(y.cpu().numpy())
                ps.append(logits.argmax(-1).cpu().numpy())
            y_all, p_all = np.concatenate(ys), np.concatenate(ps)
            out[task] = {"loss": tot / max(n, 1), "macro_f1": macro_f1(y_all, p_all, NUM_CLASSES[task]),
                         "accuracy": float((y_all == p_all).mean())}
        return out

    def _state(self, epochs_done: int) -> dict:
        return {"epoch": epochs_done, "tasks": self.tasks, "task_classes": self.model.task_classes,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(), "best": self.best, "waited": self.waited,
                "history": self.history, "rng": _rng_state()}

    # ----------------------------------------------------------- public
    def resume(self, path: Path):
        ck = load_checkpoint(Path(path), self.device)
        self.model.load_state_dict(ck["model"])
        if self.ema is not None and ck.get("ema"):
            self.ema.load_state_dict(ck["ema"])
        try:
            self.opt.load_state_dict(ck["optimizer"])
            self.sched.load_state_dict(ck["scheduler"])
        except (ValueError, KeyError) as e:
            self.log.warning(f"optimiser/scheduler not restored ({e}); continuing with fresh state")
        try:
            self.scaler.load_state_dict(ck["scaler"])
        except (KeyError, RuntimeError):
            pass
        self.start_epoch = int(ck["epoch"])
        self.best, self.waited = float(ck["best"]), int(ck["waited"])
        self.history = list(ck.get("history", []))[: self.start_epoch]
        if ck.get("rng"):
            _set_rng_state(ck["rng"])
        self.log.info(f"resumed from {path} at epoch {self.start_epoch}/{self.epochs}, best={self.best:.4f}")

    def fit(self) -> list:
        if self.start_epoch >= self.epochs:
            self.log.info(f"checkpoint already at epoch {self.start_epoch}; raise train.epochs to continue")
            return self.history
        for ep in range(self.start_epoch, self.epochs):
            t0 = time.time()
            self._set_freeze(ep)
            tr = self._train_epoch()
            va = self._validate()
            lr_head = self.opt.param_groups[0]["lr"]
            self.sched.step()
            score = sum(self.monitor[k] * va[k]["macro_f1"] for k in self.tasks)
            rec = {"epoch": ep + 1, "lr": lr_head, "score": score,
                   "train": tr, "val": va, "seconds": round(time.time() - t0, 1)}
            self.history.append(rec)
            msg = " | ".join(f"{k}: f1={va[k]['macro_f1']:.4f} acc={va[k]['accuracy']:.4f}" for k in self.tasks)
            self.log.info(f"epoch {ep + 1:03d}/{self.epochs} lr={lr_head:.2e} {msg} score={score:.4f} "
                          f"({rec['seconds']}s)")

            improved = score > self.best + self.min_delta
            if improved:
                self.best, self.waited = score, 0
            else:
                self.waited += 1
            save_checkpoint(self.ckpt_dir / "last.pth", self._state(ep + 1))
            if improved:
                save_checkpoint(self.ckpt_dir / "best.pth", self._state(ep + 1))
                self.log.info(f"  new best score {score:.4f} -> best.pth")
            write_json(self.run_dir / "history.json", self.history)
            if self.es_on and self.waited >= self.patience:
                self.log.info(f"early stopping at epoch {ep + 1}: no gain for {self.patience} epochs")
                break
        return self.history


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _rng_state() -> dict:
    import random
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}


def _set_rng_state(s: dict):
    import random
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
