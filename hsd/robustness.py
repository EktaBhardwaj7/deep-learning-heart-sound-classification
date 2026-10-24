"""Noise robustness on the test split: white, pink and 50 Hz mains hum at fixed SNRs.
Noise is added to the waveform, passed through the same band-pass, so the SNR is measured on what the model sees."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import plots
from .audio import AudioSpec
from .dataset import WindowDataset, make_loader
from .evaluate import aggregate, attach_meta, get_aggregation, predict_windows, prob_cols
from .metrics import compute_metrics
from .pipeline import data_root


def run_robustness(cfg, model, windows, spec: AudioSpec, device, out_dir: Path, logger, amp: bool) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    eval_dir = out_dir.parent / "eval"
    conditions = [("clean", None)] + [(k, s) for k in cfg["eval"]["noise_kinds"] for s in cfg["eval"]["noise_snrs_db"]]
    for task in cfg["data"]["tasks"]:
        K = len(cfg["data"]["class_names"][task])
        method = get_aggregation(eval_dir, task, cfg["eval"]["aggregation"])
        w = windows[f"{task}_test"]
        for kind, snr in conditions:
            noise = None if snr is None else (kind, float(snr))
            ds = WindowDataset(w, data_root(cfg), spec, cfg["features"], training=False, noise=noise,
                               noise_seed=int(cfg["project"]["seed"]))
            loader = make_loader(ds, 64, False, 0, int(cfg["project"]["seed"]))
            pred = attach_meta(predict_windows(model, loader, task, device, amp), w)
            rec = aggregate(pred, K, method)
            P = rec[prob_cols(K)].to_numpy()
            m = compute_metrics(rec["label"], P.argmax(1), P, K)
            shares = np.bincount(P.argmax(1), minlength=K) / len(P)
            rows.append({"task": task, "noise": kind, "snr_db": "clean" if snr is None else snr,
                         "recording_macro_f1": m["macro_f1"], "recording_accuracy": m["accuracy"],
                         "recording_auroc": m.get("auroc"), "top_class_share": float(shares.max()),
                         "predicted_share": json.dumps(np.round(shares, 3).tolist())})
            logger.info(f"[robust {task}] {kind} snr={rows[-1]['snr_db']}: F1={m['macro_f1']:.4f} "
                        f"top-class share={shares.max():.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness.csv", index=False)
    for task in cfg["data"]["tasks"]:
        d = df[df.task == task]
        clean = float(d[d.noise == "clean"]["recording_macro_f1"].iloc[0])
        curves = {}
        for kind in cfg["eval"]["noise_kinds"]:
            s = d[d.noise == kind].sort_values("snr_db", ascending=False)
            curves[kind] = (list(s.snr_db), list(s.recording_macro_f1))
        curves["clean"] = (cfg["eval"]["noise_snrs_db"][:1], [clean])
        plots.lines(curves, out_dir / f"{task}_robustness.png", "SNR (dB)", "macro-F1 (recordings)",
                    f"{task}: noise robustness (test)", ylim=(0, 1))
    return df
