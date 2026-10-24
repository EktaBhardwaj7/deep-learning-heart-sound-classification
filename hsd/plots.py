"""Figures. Every figure is saved as PNG at 150 dpi and closed, so long runs do not leak memory."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(history: list, path: Path, tasks: Sequence[str]):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, [np.mean([h["train"][t]["loss"] for t in tasks]) for h in history], label="train")
    ax[0].plot(ep, [np.mean([h["val"][t]["loss"] for t in tasks]) for h in history], label="val")
    ax[0].set(title="loss", xlabel="epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    for t in tasks:
        ax[1].plot(ep, [h["val"][t]["macro_f1"] for h in history], label=f"{t} val F1")
    ax[1].plot(ep, [h["score"] for h in history], "k:", lw=2, label="composite")
    ax[1].set(title="macro-F1 (validation)", xlabel="epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(ep, [h["lr"] for h in history]); ax[2].set_yscale("log")
    ax[2].set(title="head learning rate", xlabel="epoch"); ax[2].grid(alpha=.3)
    _save(fig, path)


def confusion(cm: np.ndarray, names: Sequence[str], path: Path, normalize: bool, title: str):
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.0 * n + 2.5),) * 2)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n), names); ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if v > (0.5 if normalize else data.max() / 2) else "black")
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def roc_pr(y_true, p1, path: Path, title: str):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y_true, p1)
    prec, rec, _ = precision_recall_curve(y_true, p1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title=f"ROC: {title}")
    ax[1].plot(rec, prec); ax[1].set(xlabel="recall", ylabel="precision", title=f"PR: {title}")
    for a in ax:
        a.grid(alpha=.3)
    _save(fig, path)


def reliability(curves: Dict[str, tuple], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, (x, y) in curves.items():
        ax.plot(x, y, "o-", label=name)
    ax.set(xlabel="confidence", ylabel="accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, path)


def lines(curves: Dict[str, tuple], path: Path, xlabel: str, ylabel: str, title: str, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, (xs, ys) in curves.items():
        ax.plot([str(x) for x in xs], ys, "o-", label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    _save(fig, path)


def gradcam_panel(spec_img, cam, occ, attn, title: str, path: Path):
    rows = 3 + (attn is not None)
    fig, ax = plt.subplots(rows, 1, figsize=(9, 2.3 * rows), sharex=True)
    ax[0].imshow(spec_img, origin="lower", aspect="auto", cmap="magma"); ax[0].set_title(title); ax[0].set_ylabel("mel")
    if cam is not None:
        ax[1].imshow(spec_img, origin="lower", aspect="auto", cmap="gray")
        ax[1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.45); ax[1].set_title("Grad-CAM")
    ax[2].imshow(occ, origin="lower", aspect="auto", cmap="hot"); ax[2].set_title("occlusion attribution")
    if attn is not None:
        ax[3].plot(attn); ax[3].set_title("attention over time"); ax[3].set_xlabel("frame")
    _save(fig, path)
