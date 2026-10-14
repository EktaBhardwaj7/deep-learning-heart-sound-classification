"""Training curves and confusion matrix figures."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch")
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch")
    axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].plot(epochs, history["train_f1"], label="train")
    axes[2].plot(epochs, history["val_f1"], label="val")
    axes[2].set_title("Macro-F1"); axes[2].set_xlabel("epoch")
    axes[2].legend(); axes[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Sequence[str],
                          out_path: str | Path,
                          normalize: bool = True) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalised)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)