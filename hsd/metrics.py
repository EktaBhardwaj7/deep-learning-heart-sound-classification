"""Classification metrics. Macro-F1 is the primary metric (classes are imbalanced)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    conf, pred = y_prob.max(1), y_prob.argmax(1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    conf, pred = y_prob.max(1), y_prob.argmax(1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(float(conf[m].mean()))
            ys.append(float(acc[m].mean()))
    return xs, ys


def compute_metrics(y_true, y_pred, y_prob=None, num_classes: Optional[int] = None) -> Dict[str, float]:
    from sklearn import metrics as M
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    K = num_classes or (y_prob.shape[1] if y_prob is not None else int(max(y_true.max(), y_pred.max())) + 1)
    labels = list(range(K))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(M.accuracy_score(y_true, y_pred)),
        "macro_precision": float(M.precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall_uar": float(M.recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(M.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(M.matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    if K == 2:
        tn, fp, fn, tp = M.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update(sensitivity=float(tp / max(tp + fn, 1)), specificity=float(tn / max(tn + fp, 1)),
                   ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)))
    if y_prob is not None:
        p = np.asarray(y_prob, dtype=np.float64)
        out["ece"] = expected_calibration_error(y_true, p)
        out["brier"] = float(np.mean(np.sum((p - np.eye(K)[y_true]) ** 2, axis=1)))
        try:
            if K == 2:
                out["auroc"] = float(M.roc_auc_score(y_true, p[:, 1]))
                out["auprc"] = float(M.average_precision_score(y_true, p[:, 1]))
            else:
                out["auroc"] = float(M.roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=labels))
        except ValueError:                         # only one class present in y_true
            out["auroc"] = float("nan")
    return out


def bootstrap_macro_f1_ci(y_true, y_pred, clusters, num_classes: int, n: int = 1000, seed: int = 0,
                          alpha: float = 0.05):
    """Cluster bootstrap: whole patients (or recordings) are resampled, not single windows."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y_true, y_pred, clusters = np.asarray(y_true), np.asarray(y_pred), np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[p] for p in pick])
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=list(range(num_classes)),
                             average="macro", zero_division=0))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
