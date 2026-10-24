"""Glue between configuration, the index, the splits and the data loaders. Used by every CLI command."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .audio import AudioSpec, spec_from_cfg
from .config import run_paths
from .dataset import WindowDataset, make_loader
from .labels import build_circor_index, build_yaseen_index, expand_windows
from .splits import load_or_create_splits

NUM_CLASSES = {"yaseen": 5, "circor": 2}


def data_root(cfg: dict) -> Path:
    return Path(cfg["data"]["root"])


def prepare_index(cfg: dict, logger) -> pd.DataFrame:
    """Builds the recording index from the raw folders, then the leakage-free splits (cached in runs/<name>/)."""
    root = data_root(cfg)
    d = cfg["data"]
    parts = []
    if "yaseen" in d["tasks"]:
        logger.info("indexing Yaseen recordings")
        parts.append(build_yaseen_index(root, d["yaseen_dir"], d["class_names"]["yaseen"],
                                        float(d["min_duration_seconds"]), logger))
    if "circor" in d["tasks"]:
        logger.info("indexing CirCor recordings and metadata")
        parts.append(build_circor_index(root, d["circor_dir"], root / d["circor_meta"],
                                        d["nonmurmur_site_policy"], float(d["min_duration_seconds"]), logger))
    index = pd.concat(parts, ignore_index=True)
    logger.info("index status counts:\n" + index.groupby(["dataset", "status"]).size().to_string())
    return load_or_create_splits(index, cfg, run_paths(cfg)["run"], logger)


def build_windows(cfg: dict, idx: pd.DataFrame, spec: AudioSpec) -> Dict[str, pd.DataFrame]:
    """Returns {'<task>_<split>': window table}. Only status == ok recordings are used for train/val/test."""
    seg = float(cfg["data"]["segment_seconds"])
    hop = float(cfg["data"]["circor_hop_seconds"])
    out = {}
    for task in cfg["data"]["tasks"]:
        sub = idx[(idx.dataset == task) & (idx.status == "ok")]
        for split in ("train", "val", "test"):
            rows = sub[sub.split == split]
            if rows.empty:
                raise RuntimeError(f"no usable recordings for {task} {split}; check the splits")
            hop_s = None if task == "yaseen" else hop
            w = expand_windows(rows, seg, hop_s)
            out[f"{task}_{split}"] = w
    # Unknown-murmur patients: never trained on, used only for the uncertainty analysis
    if "circor" in cfg["data"]["tasks"]:
        held = idx[(idx.dataset == "circor") & (idx.split == "unknown") & (idx.status == "unknown")]
        if len(held):
            out["circor_unknown"] = expand_windows(held, seg, hop)
    # row ids are global so predictions can be joined back to their windows
    offset = 0
    for k in sorted(out):
        out[k] = out[k].reset_index(drop=True)
        out[k]["row_id"] = np.arange(offset, offset + len(out[k]))
        offset += len(out[k])
    return out


def build_loaders(cfg: dict, windows: Dict[str, pd.DataFrame], spec: AudioSpec, device=None):
    """Loaders per task and split. Unknown-murmur windows are added as split 'unknown' for uncertainty analysis."""
    root = data_root(cfg)
    f = cfg["features"]
    bs = cfg["train"]["batch_size"]
    nw = int(cfg["train"].get("num_workers", 0))
    seed = int(cfg["project"]["seed"])
    loaders = {}
    for task in cfg["data"]["tasks"]:
        loaders[task] = {}
        for split in ("train", "val", "test"):
            w = windows[f"{task}_{split}"]
            ds = WindowDataset(w, root, spec, f, training=(split == "train"))
            loaders[task][split] = make_loader(ds, int(bs[task]), split == "train", nw, seed)
    return loaders


def class_weights(cfg: dict, windows: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    out = {}
    if cfg["train"].get("class_weights", "balanced") != "balanced":
        return {t: None for t in cfg["data"]["tasks"]}
    for task in cfg["data"]["tasks"]:
        y = windows[f"{task}_train"]["label"].to_numpy()
        K = NUM_CLASSES[task]
        counts = np.maximum(np.bincount(y, minlength=K).astype(float), 1.0)
        w = counts.sum() / (K * counts)
        out[task] = (w / w.mean()).tolist()
    return out


def spec_and_device(cfg: dict):
    from .utils import pick_device
    return spec_from_cfg(cfg), pick_device(cfg["project"].get("device", "auto"))
