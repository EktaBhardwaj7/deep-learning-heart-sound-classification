"""Uncertainty analysis (RQ4).

- Monte Carlo dropout: `mc_samples` stochastic passes per window (dropout on, BatchNorm fixed).
- Recording-level mean probability and entropy, aggregated with the rule chosen on validation.
- Risk-coverage: keep the most confident x % of recordings and measure the error rate that remains.
- Three triage zones (CirCor): discharge (confidently Absent), refer (confidently Present), specialist (the rest).
  Zone thresholds are chosen ON VALIDATION to hit NPV >= npv_target (discharge) and PPV >= ppv_target (refer).
- 'Unknown' murmur windows (never used for training) should carry more uncertainty than known test windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .evaluate import aggregate, attach_meta, get_aggregation, predict_windows, prob_cols
from .pipeline import build_loaders
from .utils import write_json


def _risk_coverage(y, pred, conf):
    order = np.argsort(-conf)
    correct = (pred == y)[order]
    n = len(y)
    out = {}
    for c in (1.0, 0.9, 0.8, 0.7, 0.5):
        k = max(1, int(round(c * n)))
        out[f"{int(c * 100)}%"] = float(1 - correct[:k].mean())
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(correct) / np.arange(1, n + 1)
    step = max(1, n // 20)
    return out, (list(np.round(cov[::step] * 100).astype(int)), list(risk[::step]))


def _choose_zone_thresholds(rec_val: pd.DataFrame, npv_target: float, ppv_target: float, min_n: int):
    y = rec_val["label"].to_numpy()
    p_absent, p_present = rec_val["p0"].to_numpy(), rec_val["p1"].to_numpy()
    grid = np.round(np.arange(0.50, 0.995, 0.005), 3)
    disc = None
    for t in grid:
        m = p_absent >= t
        if m.sum() >= min_n and (y[m] == 0).mean() >= npv_target:
            disc = float(t)
            break
    refer = None
    for t in grid[::-1]:
        m = p_present >= t
        if m.sum() >= min_n and (y[m] == 1).mean() >= ppv_target:
            refer = float(t)
            break
    return disc, refer


def run_uncertainty(cfg, model, loaders, windows, spec, device, out_dir: Path, logger, amp: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    S = int(cfg["eval"].get("mc_samples", 30))
    tri_cfg = cfg["eval"]["triage"]
    summary = {"mc_samples": S}
    eval_dir = out_dir.parent / "eval"
    for task in cfg["data"]["tasks"]:
        names = cfg["data"]["class_names"][task]
        K = len(names)
        method = get_aggregation(eval_dir, task, cfg["eval"]["aggregation"])
        val = attach_meta(predict_windows(model, loaders[task]["val"], task, device, amp, mc_samples=S),
                          windows[f"{task}_val"])
        test = attach_meta(predict_windows(model, loaders[task]["test"], task, device, amp, mc_samples=S),
                           windows[f"{task}_test"])
        rec_val = aggregate(val, K, method)
        rec_test = aggregate(test, K, method)
        ent = test.groupby("rec_id")[["pred_entropy", "mutual_info"]].mean().reset_index()
        rec_test = rec_test.merge(ent, on="rec_id")
        P = rec_test[prob_cols(K)].to_numpy()
        y, pred, conf = rec_test["label"].to_numpy(), P.argmax(1), P.max(1)
        rc, curve = _risk_coverage(y, pred, conf)
        plots.lines({task: curve}, out_dir / f"{task}_risk_coverage.png", "coverage (%)", "error rate",
                    f"{task}: selective prediction (test)", ylim=(0, 1))
        t_summary = {
            "aggregation": method,
            "risk_at_coverage": rc,
            "mean_entropy_correct": float(rec_test.loc[pred == y, "pred_entropy"].mean()) if (pred == y).any() else None,
            "mean_entropy_incorrect": float(rec_test.loc[pred != y, "pred_entropy"].mean()) if (pred != y).any() else None,
        }
        if K == 2:
            rv = rec_val.merge(val.groupby("rec_id")[["pred_entropy"]].mean().reset_index(), on="rec_id")
            d_thr, r_thr = _choose_zone_thresholds(rec_val, float(tri_cfg["npv_target"]),
                                                   float(tri_cfg["ppv_target"]), int(tri_cfg["min_recordings"]))
            fallback = {"discharge": 0.95, "refer": 0.80}
            chosen = {"discharge": d_thr if d_thr is not None else fallback["discharge"],
                      "refer": r_thr if r_thr is not None else fallback["refer"]}
            t_summary["zone_thresholds"] = {**chosen, "source": "validation" if (d_thr and r_thr) else
                                            "fallback (validation target not reached)"}
            p0, p1 = rec_test["p0"].to_numpy(), rec_test["p1"].to_numpy()
            discharge = p0 >= chosen["discharge"]
            refer = (p1 >= chosen["refer"]) & ~discharge
            specialist = ~(discharge | refer)
            zones = {}
            for name, m in (("discharge", discharge), ("refer", refer), ("specialist", specialist)):
                n = int(m.sum())
                zones[name] = {"share": round(n / len(y), 4), "n": n,
                               "present_rate": round(float(y[m].mean()), 4) if n else None}
            t_summary["zones"] = zones
            t_summary["zone_accuracy_non_specialist"] = (
                float((np.where(discharge, 0, np.where(refer, 1, -1))[~specialist] == y[~specialist]).mean())
                if (~specialist).any() else None)
        summary[task] = t_summary
        rec_test.to_csv(out_dir / f"{task}_recordings_uncertainty.csv", index=False)
        logger.info(f"[{task}] risk@coverage {rc} | entropy correct={t_summary['mean_entropy_correct']}")

    if "circor" in cfg["data"]["tasks"] and "circor_unknown" in windows and len(windows["circor_unknown"]):
        from .dataset import WindowDataset, make_loader
        from .pipeline import data_root
        ds = WindowDataset(windows["circor_unknown"], data_root(cfg), spec, cfg["features"], training=False)
        loader = make_loader(ds, 64, False, 0, int(cfg["project"]["seed"]))
        unk = predict_windows(model, loader, "circor", device, amp, mc_samples=S)
        known = attach_meta(predict_windows(model, loaders["circor"]["test"], "circor", device, amp, mc_samples=S),
                            windows["circor_test"])
        summary["unknown_vs_known_windows"] = {
            "unknown_n": int(len(unk)),
            "unknown_mean_entropy": float(unk["pred_entropy"].mean()),
            "known_test_mean_entropy": float(known["pred_entropy"].mean()),
            "unknown_mean_mutual_info": float(unk["mutual_info"].mean()),
            "known_test_mean_mutual_info": float(known["mutual_info"].mean()),
        }
        logger.info("Unknown CirCor windows: entropy "
                    f"{summary['unknown_vs_known_windows']['unknown_mean_entropy']:.3f} vs known "
                    f"{summary['unknown_vs_known_windows']['known_test_mean_entropy']:.3f}")
    summary_path = out_dir / "uncertainty_summary.json"
    write_json(summary_path, summary)
    logger.info(f"uncertainty written to {out_dir}")
    return summary
