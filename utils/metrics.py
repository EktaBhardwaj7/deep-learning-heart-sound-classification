"""Evaluation metrics. Macro-F1 is the primary reported metric."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None,
                    class_names: Sequence[str] | None = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if y_prob is not None:
        try:
            out["macro_roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["macro_roc_auc"] = float("nan")
    return out


def per_class_report(y_true, y_pred, class_names=None) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=list(class_names) if class_names else None,
        digits=4, zero_division=0,
    )


def confusion(y_true, y_pred, num_classes=None) -> np.ndarray:
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))