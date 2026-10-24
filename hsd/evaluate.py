"""Turns window-level probabilities into recording-level decisions, with every choice made on validation data.

Rules that are fixed before looking at the test set:
  * the rule for combining windows into a recording (mean / max / median / noisy-OR / vote) is chosen on validation
  * the CirCor operating threshold is chosen on validation (>= min_sensitivity, highest precision)
  * temperature scaling is fitted on validation, and only if there are enough validation windows
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import plots
from .metrics import bootstrap_macro_f1_ci, compute_metrics, expected_calibration_error, reliability_curve
from .utils import progress, write_json

CANDIDATES = {2: ["mean", "max", "median", "noisy_or"], 5: ["mean", "max", "median", "vote"]}


def prob_cols(K: int) -> List[str]:
    return [f"p{k}" for k in range(K)]


# ------------------------------------------------------------------ prediction
@torch.no_grad()
def predict_windows(model, loader, task: str, device, amp: bool = False, mc_samples: int = 0,
                    tta_shifts: Optional[List[int]] = None) -> pd.DataFrame:
    """Window-level class probabilities. mc_samples > 0 keeps dropout on and returns uncertainty columns."""
    K = model.task_classes[task]
    if mc_samples > 0:
        model.eval()
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout1d)):
                m.train()
    else:
        model.eval()
    rows_id, labels, probs, ent_pred, ent_exp = [], [], [], [], []
    for x, y, rid in progress(loader, desc=f"predict[{task}]", leave=False, ncols=88):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            if mc_samples > 0:
                ps = torch.stack([torch.softmax(model(x, task=task).float(), -1) for _ in range(mc_samples)])
                mean = ps.mean(0)
                eps = -(ps * torch.log(ps.clamp_min(1e-12))).sum(-1).mean(0)
                ent_exp.append(eps.cpu().numpy())
            elif tta_shifts:
                mean = torch.stack([torch.softmax(model(torch.roll(x, s, dims=-1), task=task).float(), -1)
                                    for s in tta_shifts]).mean(0)
            else:
                mean = torch.softmax(model(x, task=task).float(), -1)
        probs.append(mean.float().cpu().numpy())
        rows_id.append(np.asarray(rid))
        labels.append(np.asarray(y))
        if mc_samples > 0:
            ent_pred.append(-(mean * torch.log(mean.clamp_min(1e-12))).sum(-1).cpu().numpy())
    P = np.concatenate(probs)
    df = pd.DataFrame({"row_id": np.concatenate(rows_id), "label": np.concatenate(labels)})
    for k in range(K):
        df[f"p{k}"] = P[:, k]
    if mc_samples > 0:
        df["pred_entropy"] = np.concatenate(ent_pred)
        df["expected_entropy"] = np.concatenate(ent_exp)
        df["mutual_info"] = df["pred_entropy"] - df["expected_entropy"]
    return df


def attach_meta(pred: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    meta = windows.set_index("row_id")[["path", "rec_id", "patient_id", "start"]]
    return pred.join(meta, on="row_id")


# ------------------------------------------------------------------ aggregation
def aggregate(win: pd.DataFrame, K: int, method: str) -> pd.DataFrame:
    pc = prob_cols(K)
    rows = []
    for rec_id, g in win.groupby("rec_id", sort=True):
        P = g[pc].to_numpy(dtype=np.float64)
        if method == "noisy_or" and K == 2:
            p1 = 1.0 - float(np.prod(1.0 - np.clip(P[:, 1], 0.0, 1.0 - 1e-9)))
            agg = np.array([1.0 - p1, p1])
        elif method == "max":
            agg = P.max(0)
            agg = agg / agg.sum()
        elif method == "median":
            agg = np.median(P, axis=0)
            agg = agg / agg.sum()
        elif method == "vote":
            agg = np.bincount(P.argmax(1), minlength=K) / len(P)
        else:
            agg = P.mean(0)
        row = {"rec_id": rec_id, "path": g["path"].iloc[0], "label": int(g["label"].iloc[0]),
               "patient_id": g["patient_id"].iloc[0], "n_windows": len(g)}
        row.update({pc[k]: float(agg[k]) for k in range(K)})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_aggregation(val_win: pd.DataFrame, K: int, requested: str, logger, out_file: Path) -> str:
    if requested != "auto":
        choice = {"method": requested, "source": "config"}
    else:
        scores = {}
        for m in CANDIDATES[K]:
            rec = aggregate(val_win, K, m)
            P = rec[prob_cols(K)].to_numpy()
            y = rec["label"].to_numpy()
            if K == 2:
                thr = threshold_best_f1(y, P[:, 1])
                pred = (P[:, 1] >= thr).astype(int)
            else:
                pred = P.argmax(1)
            scores[m] = compute_metrics(y, pred, P, K)["macro_f1"]
        best = max(scores, key=scores.get)
        choice = {"method": best, "source": "validation", "validation_macro_f1": scores}
        logger.info("aggregation chosen on validation: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
                    + f"  -> {best}")
    write_json(out_file, choice)
    return choice["method"]


def get_aggregation(run_eval_dir: Path, task: str, cfg_value: str) -> str:
    f = run_eval_dir / f"aggregation_{task}.json"
    if f.exists():
        return json.loads(f.read_text())["method"]
    return "mean" if cfg_value == "auto" else cfg_value


# ------------------------------------------------------------------ thresholds, calibration
def threshold_best_f1(y, p1) -> float:
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p1)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def threshold_for_sensitivity(y, p1, min_sens: float) -> Optional[float]:
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p1)
    idx = np.where(rec[:-1] >= min_sens)[0]
    if len(idx) == 0:
        return None
    best = prec[idx].max()
    cand = idx[prec[idx] >= best - 1e-12]
    return float(thr[cand[-1]])


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    logits = torch.log(torch.tensor(P, dtype=torch.float32).clamp_min(1e-8))
    target = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), target)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(P, 1e-8, 1.0)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ per task evaluation
def evaluate_task(task: str, model, loaders: dict, windows: Dict[str, pd.DataFrame], cfg: dict, device,
                  out_dir: Path, logger, amp: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = cfg["data"]["class_names"][task]
    K = len(names)
    ev = cfg["eval"]
    val = attach_meta(predict_windows(model, loaders[task]["val"], task, device, amp), windows[f"{task}_val"])
    test = attach_meta(predict_windows(model, loaders[task]["test"], task, device, amp), windows[f"{task}_test"])

    method = choose_aggregation(val, K, ev["aggregation"], logger, out_dir / f"aggregation_{task}.json")
    rec_val = aggregate(val, K, method)
    rec_test = aggregate(test, K, method)
    Pv, Pt = rec_val[prob_cols(K)].to_numpy(), rec_test[prob_cols(K)].to_numpy()
    yv, yt = rec_val["label"].to_numpy(), rec_test["label"].to_numpy()

    res = {"task": task, "aggregation": method,
           "aggregation_source": "validation" if ev["aggregation"] == "auto" else "config",
           "segment": compute_metrics(test["label"], test[prob_cols(K)].to_numpy().argmax(1),
                                      test[prob_cols(K)].to_numpy(), K),
           "recording": compute_metrics(yt, Pt.argmax(1), Pt, K)}
    clusters = rec_test["patient_id"].to_numpy() if task == "circor" else rec_test["rec_id"].to_numpy()
    lo, hi = bootstrap_macro_f1_ci(yt, Pt.argmax(1), clusters, K, n=int(ev.get("bootstrap", 1000)))
    res["recording"]["macro_f1_ci95"] = [lo, hi]
    res["recording"]["ci_clusters"] = "patient" if task == "circor" else "recording"

    if K == 2:
        thr_s = threshold_for_sensitivity(yv, Pv[:, 1], float(ev["min_sensitivity"]))
        thr_f = threshold_best_f1(yv, Pv[:, 1])
        for key, thr in (("operating_sensitivity", thr_s), ("operating_best_f1", thr_f)):
            if thr is None:
                res[key] = {"note": f"no validation threshold reaches {ev['min_sensitivity']} sensitivity"}
                continue
            pred = (Pt[:, 1] >= thr).astype(int)
            m = compute_metrics(yt, pred, Pt, K)
            m["threshold"] = thr
            m["threshold_chosen_on"] = "validation"
            res[key] = m
        plots.roc_pr(yt, Pt[:, 1], out_dir / f"{task}_roc_pr.png", f"{task}, recording level")

    # calibration: temperature only with enough validation windows
    n_val = len(val)
    if n_val >= int(ev["temperature_min_windows"]):
        T = fit_temperature(val[prob_cols(K)].to_numpy(), val["label"].to_numpy())
        Ptc = apply_temperature(test[prob_cols(K)].to_numpy(), T)
        cal = {"temperature": T, "validation_windows": n_val}
    else:
        T, Ptc = None, test[prob_cols(K)].to_numpy()
        cal = {"temperature": None, "validation_windows": n_val,
               "skipped": f"only {n_val} validation windows (< {ev['temperature_min_windows']})"}
    yw = test["label"].to_numpy()
    Pw = test[prob_cols(K)].to_numpy()
    cal["ece_before"] = expected_calibration_error(yw, Pw)
    cal["ece_after"] = expected_calibration_error(yw, Ptc) if T is not None else None
    res["calibration"] = cal

    plots.confusion(_cm(yt, Pt.argmax(1), K), names,
                    out_dir / f"{task}_confusion.png", True, f"{task} recording level (row-normalised)")
    plots.confusion(_cm(yt, Pt.argmax(1), K), names, out_dir / f"{task}_confusion_counts.png", False,
                    f"{task} recording level (counts)")
    plots.reliability({"raw": reliability_curve(yw, Pw), **({"temperature-scaled": reliability_curve(yw, Ptc)}
                       if T is not None else {})}, out_dir / f"{task}_reliability.png", f"{task} reliability (windows)")
    from sklearn.metrics import classification_report
    (out_dir / f"{task}_report.txt").write_text(classification_report(
        yt, Pt.argmax(1), labels=list(range(K)), target_names=names, digits=4, zero_division=0))
    write_json(out_dir / f"{task}_metrics.json", res)
    test.to_csv(out_dir / f"{task}_windows.csv", index=False)
    rec_test.to_csv(out_dir / f"{task}_recordings.csv", index=False)
    r = res["recording"]
    logger.info(f"[{task}] windows F1={res['segment']['macro_f1']:.4f} | recordings ({method}) "
                f"F1={r['macro_f1']:.4f} [{lo:.3f},{hi:.3f}] AUROC={r.get('auroc', float('nan')):.4f}")
    if K == 2 and "sensitivity" in res.get("operating_sensitivity", {}):
        o = res["operating_sensitivity"]
        logger.info(f"[{task}] operating point from validation (thr={o['threshold']:.3f}): "
                    f"sensitivity={o['sensitivity']:.3f} specificity={o['specificity']:.3f}")
    return res


def _cm(y, p, K):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y, p, labels=list(range(K)))
