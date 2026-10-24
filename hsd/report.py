"""Writes runs/<name>/REPORT.md from the JSON and CSV files of a run. Nothing is typed by hand."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build_report(cfg: dict, paths: dict, out: Path) -> Path:
    L = [f"# Results: {cfg['project']['name']}", ""]
    L += [f"Model `{cfg['model']['name']}` (pooling {cfg['model'].get('pooling', 'n/a')}) | input "
          f"`{cfg['features']['representation']}` | tasks {cfg['data']['tasks']} | seed {cfg['project']['seed']}", ""]
    summ = _j(paths["eval"] / "summary.json")
    if summ:
        for task, r in summ["tasks"].items():
            rec, seg = r["recording"], r["segment"]
            ci = rec["macro_f1_ci95"]
            L += [f"## {task}", "",
                  f"Recording rule: **{r['aggregation']}** (chosen on {r['aggregation_source']}).", "",
                  "| metric | windows (test) | recordings (test) |", "|---|---|---|",
                  f"| accuracy | {seg['accuracy']:.4f} | {rec['accuracy']:.4f} |",
                  f"| macro-F1 | {seg['macro_f1']:.4f} | {rec['macro_f1']:.4f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}, "
                  f"{rec['ci_clusters']} bootstrap) |",
                  f"| AUROC | {seg.get('auroc', float('nan')):.4f} | {rec.get('auroc', float('nan')):.4f} |"]
            if "sensitivity" in r.get("operating_sensitivity", {}):
                o = r["operating_sensitivity"]
                L.append(f"| sensitivity at validation-chosen threshold {o['threshold']:.3f} | | {o['sensitivity']:.4f} |")
                L.append(f"| specificity at the same threshold | | {o['specificity']:.4f} |")
            c = r["calibration"]
            if c.get("temperature") is not None:
                L.append(f"| ECE before / after temperature scaling | | {c['ece_before']:.3f} / {c['ece_after']:.3f} |")
            else:
                L.append(f"| ECE | | {c['ece_before']:.3f} (temperature scaling skipped: {c['skipped']}) |")
            L.append("")
    rb = paths["robust"] / "robustness.csv"
    if rb.exists():
        d = pd.read_csv(rb)
        for task in d.task.unique():
            pv = d[d.task == task].pivot_table(index="noise", columns="snr_db", values="recording_macro_f1",
                                                aggfunc="first").round(3)
            L += [f"## Noise robustness: {task} (recording macro-F1)", "", "```", pv.to_string(), "```", ""]
    un = _j(paths["uncert"] / "uncertainty_summary.json")
    if un:
        L += ["## Uncertainty", "", "```json", json.dumps(un, indent=2), "```", ""]
    ex = _j(paths["explain"] / "explain_summary.json")
    if ex:
        L += ["## Explanations", "", "```json", json.dumps(ex, indent=2), "```", ""]
    hist = _j(paths["run"] / "history.json")
    if hist:
        best = max(hist, key=lambda h: h["score"])
        L += [f"## Training", "", f"{len(hist)} epochs; best composite score {best['score']:.4f} at epoch "
              f"{best['epoch']}.", ""]
    out.write_text("\n".join(L), encoding="utf-8")
    return out
