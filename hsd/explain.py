"""Explanations (RQ5).

Grad-CAM (2-D models), occlusion attribution (model-agnostic, stands in for SHAP in phase 1),
a deletion test for faithfulness, attention maps (temporal models), and a CirCor cardiac-phase check.

Phase check: CirCor .tsv segmentation files give S1 / systole / S2 / diastole. ASSUMED state codes:
1 = S1, 2 = systole, 3 = S2, 4 = diastole (PhysioNet 2022 convention). Verify against the dataset readme.
The .tsv does not mark murmur extents, so this check describes cardiac phase only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import plots
from .dataset import WindowDataset
from .evaluate import predict_windows
from .pipeline import data_root
from .utils import write_json

PHASES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def _probs(model, x, task):
    with torch.no_grad():
        return torch.softmax(model(x, task=task).float(), -1)[0]


def gradcam(model, x: torch.Tensor, task: str, cls: int) -> Optional[np.ndarray]:
    layer = model.cam_layer
    if layer is None:
        return None
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(x, task=task)
            a = store["a"]
            g = torch.autograd.grad(logits[0, cls], a)[0]
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    finally:
        h.remove()
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)


@torch.no_grad()
def occlusion(model, x: torch.Tensor, task: str, cls: int, bf: int = 8, bt: int = 10, batch: int = 64) -> np.ndarray:
    base = _probs(model, x, task)[cls].item()
    Fd, Td = x.shape[-2:]
    crops, cells = [], []
    for f0 in range(0, Fd, bf):
        for t0 in range(0, Td, bt):
            x2 = x.clone()
            x2[..., f0:f0 + bf, t0:t0 + bt] = 0.0          # 0 = mean after standardisation
            crops.append(x2)
            cells.append((f0, t0))
    drops = []
    for i in range(0, len(crops), batch):
        xb = torch.cat(crops[i:i + batch], 0)
        drops.append(base - torch.softmax(model(xb, task=task).float(), -1)[:, cls])
    drops = torch.cat(drops).cpu().numpy()
    heat = np.zeros((Fd, Td), dtype=np.float32)
    for (f0, t0), d in zip(cells, drops):
        heat[f0:f0 + bf, t0:t0 + bt] = d
    heat = np.clip(heat, 0, None)
    return heat / (heat.max() + 1e-9)


@torch.no_grad()
def deletion_test(model, x: torch.Tensor, task: str, cls: int, heat: np.ndarray, fracs, seeds: int = 5) -> dict:
    """Remove the top-attributed pixels vs the same number of random pixels. Faithful maps remove a larger drop."""
    base = _probs(model, x, task)[cls].item()
    n = heat.size
    order = np.argsort(-heat.ravel())
    rng = np.random.default_rng(0)
    out = {}
    for f in fracs:
        k = max(1, int(f * n))
        xm = x.clone().view(-1)
        xm[torch.as_tensor(order[:k].copy())] = 0.0
        top = base - _probs(model, xm.view_as(x), task)[cls].item()
        rnd = []
        for _ in range(seeds):
            xr = x.clone().view(-1)
            xr[torch.as_tensor(rng.choice(n, size=k, replace=False))] = 0.0
            rnd.append(base - _probs(model, xr.view_as(x), task)[cls].item())
        out[f"{int(f * 100)}%"] = {"attributed": float(top), "random": float(np.mean(rnd))}
    return out


def attention_profile(model, x: torch.Tensor, task: str, T: int) -> Optional[np.ndarray]:
    pool = getattr(model.backbone, "pool", None)
    if pool is None or getattr(pool, "last_weights", None) is None:
        return None
    model.eval()
    with torch.no_grad():
        model(x, task=task)
    w = pool.last_weights[0].float().view(1, 1, -1)
    prof = F.interpolate(w, size=T, mode="linear", align_corners=False)[0, 0].cpu().numpy()
    return (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)


def load_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["start", "end", "state"]
    return df.astype(float)


def phase_shares(attr_frames: np.ndarray, tsv: pd.DataFrame, window_start: float, sr: int, hop: int,
                 top_frac: float = 0.30) -> Optional[dict]:
    T = len(attr_frames)
    t = window_start + np.arange(T) * hop / sr
    state = np.zeros(T, dtype=int)
    for _, r in tsv.iterrows():
        state[(t >= r["start"]) & (t < r["end"])] = int(r["state"])
    k = max(1, int(top_frac * T))
    top = np.argsort(-attr_frames)[:k]
    if (state[top] == 0).all():
        return None
    out = {name: float(np.mean(state[top] == code)) for code, name in PHASES.items()}
    out["unlabelled"] = float(np.mean(state[top] == 0))
    return out


def _deletion_row(v) -> dict:
    a = float(np.mean([d["attributed"] for d in v]))
    r = float(np.mean([d["random"] for d in v]))
    ratio = a / r if r > 1e-6 else None      # undefined when removing random regions does not hurt
    return {"attributed_drop": a, "random_drop": r, "ratio": ratio}


def run_explain(cfg, model, windows, spec, device, out_dir: Path, logger) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.representation != "logmel":
        logger.info("explanations skipped: raw-waveform models have no time-frequency map")
        return {"skipped": "raw waveform"}
    ex = cfg["eval"]["explain"]
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    root = data_root(cfg)
    summary = {}
    for task in cfg["data"]["tasks"]:
        w = windows[f"{task}_test"]
        K = len(cfg["data"]["class_names"][task])
        per = max(1, int(ex["max_samples"]) // K)
        chosen = []
        for k in range(K):
            pool = np.where(w["label"].to_numpy() == k)[0]
            if len(pool):
                chosen += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
        ds = WindowDataset(w, root, spec, cfg["features"], training=False)
        recs, dele, timing, attn_corr = [], {}, [], []
        model.eval()
        for n, i in enumerate(chosen):
            x, y, _ = ds[int(i)]
            x = x.unsqueeze(0).to(device)
            p = _probs(model, x, task).cpu().numpy()
            cls = int(p.argmax())
            cam = gradcam(model, x, task, cls)
            occ = occlusion(model, x, task, cls)
            for k_, v in deletion_test(model, x, task, cls, occ, ex["deletion_fractions"]).items():
                dele.setdefault(k_, []).append(v)
            attn = attention_profile(model, x, task, x.shape[-1])
            if attn is not None:
                a, b = attn, occ.mean(0)
                attn_corr.append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 and a.std() > 0 else 0.0)
            frames = cam.mean(0) if cam is not None else occ.mean(0)
            if task == "circor":
                row = w.iloc[int(i)]
                tsv = load_tsv((root / row["path"]).with_suffix(".tsv"))
                if tsv is not None:
                    sh = phase_shares(frames, tsv, float(row["start"]), spec.sr, spec.hop)
                    if sh:
                        timing.append(sh)
            recs.append({"path": w.iloc[int(i)]["path"], "start": float(w.iloc[int(i)]["start"]),
                         "label": int(y), "pred": cls, "p_pred": float(p[cls]),
                         "attention_vs_occlusion_r": attn_corr[-1] if attn is not None else None})
            if n < int(ex["figures"]):
                plots.gradcam_panel(x[0, 0].cpu().numpy(), cam, occ, attn,
                                    f"{task} | true={int(y)} pred={cls} p={p[cls]:.2f}",
                                    out_dir / f"{task}_example_{n:02d}.png")
        summary[task] = {
            "n_explained": len(recs),
            "accuracy_on_explained": float(np.mean([r["label"] == r["pred"] for r in recs])) if recs else None,
            "deletion_faithfulness": {
                k: {"attributed_drop": float(np.mean([d["attributed"] for d in v])),
                    "random_drop": float(np.mean([d["random"] for d in v])),
                    "ratio": float(np.mean([d["attributed"] for d in v]) / (np.mean([d["random"] for d in v]) + 1e-9))}
                for k, v in dele.items()},
            "attention_vs_occlusion_r_mean": float(np.mean(attn_corr)) if attn_corr else None,
            "circor_phase_share_of_top_frames": (
                {k: float(np.mean([d[k] for d in timing])) for k in timing[0]} if timing else
                "no .tsv segmentation found for the explained windows"),
        }
        pd.DataFrame(recs).to_csv(out_dir / f"{task}_explain_samples.csv", index=False)
        logger.info(f"[{task}] explain: n={len(recs)} deletion ratio(10%)="
                    f"{summary[task]['deletion_faithfulness'].get('10%', {}).get('ratio')}")
    write_json(out_dir / "explain_summary.json", summary)
    return summary
