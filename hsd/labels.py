"""Build the recording index for both datasets, following the folder layout of the source data.

Yaseen   data/heart_sound_dataset/<CLASS>/**/*.wav          (8 kHz, no patient IDs)
CirCor   data/circor/**/<patient>_<site>.wav                 (4 kHz; metadata in data/circor/training_data.csv)

Each recording gets one row. Windows are generated later by `expand_windows`.
Every path is stored relative to `data.root`, so the index stays valid if the project moves.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf

SITES = ("AV", "PV", "TV", "MV")
PATIENT_STEM = re.compile(r"^(?P<pid>\d+)_(?P<site>AV|PV|TV|MV)$", re.IGNORECASE)

INDEX_COLUMNS = ["dataset", "path", "rec_id", "patient_id", "site", "label", "class_name",
                 "status", "duration", "rms", "clip_frac", "patient_murmur"]


def _relpath(p: Path, root: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()


def _quality(y: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        return 0.0, 1.0
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    clip = float(np.mean(np.abs(y) >= 0.999 * peak)) if peak > 0 else 1.0
    return rms, clip


def build_yaseen_index(root: Path, yaseen_dir: str, class_names: List[str], min_duration: float,
                       log=None) -> pd.DataFrame:
    base = root / yaseen_dir
    rows = []
    for cls in class_names:
        folder = base / cls
        if not folder.exists():
            raise FileNotFoundError(f"Yaseen class folder missing: {folder}")
        for wav in sorted(folder.rglob("*.wav")):
            rel = _relpath(wav, root)
            status, dur, rms, clip = "ok", 0.0, 0.0, 0.0
            try:
                y, sr = sf.read(str(wav), always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                dur = len(y) / sr
                rms, clip = _quality(y)
                if dur < min_duration:
                    status = "too_short"
                elif rms < 1e-6:
                    status = "silent"
            except Exception as e:  # unreadable container or codec
                status = "unreadable"
                if log:
                    log.warning(f"unreadable {rel}: {e}")
            rows.append({"dataset": "yaseen", "path": rel, "rec_id": rel, "patient_id": rel, "site": "",
                         "label": class_names.index(cls), "class_name": cls, "status": status,
                         "duration": round(dur, 3), "rms": rms, "clip_frac": clip, "patient_murmur": ""})
    if not rows:
        raise RuntimeError(f"no Yaseen .wav files found under {base}")
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", c.strip().lower().rstrip(":")).strip()


def read_circor_metadata(meta_csv: Path) -> Dict[str, dict]:
    meta = pd.read_csv(meta_csv, dtype=str)
    cols = {_norm_col(c): c for c in meta.columns}
    for need in ("patient id", "murmur", "murmur locations"):
        if need not in cols:
            raise KeyError(f"'{need}' column not found in {meta_csv}; columns = {list(meta.columns)}")
    out = {}
    for _, r in meta.iterrows():
        pid = str(r[cols["patient id"]]).strip()
        mur = str(r[cols["murmur"]]).strip()
        raw_locs = r[cols["murmur locations"]]
        locs = set()
        if isinstance(raw_locs, str) and raw_locs.strip().lower() not in ("", "nan", "none"):
            locs = {x.strip().upper() for x in raw_locs.split("+") if x.strip()}
        out[pid] = {"murmur": mur, "locs": locs}
    return out


def find_circor_wavs(root: Path, circor_dir: str) -> Dict[str, Path]:
    base = root / circor_dir
    found: Dict[str, Path] = {}
    for wav in sorted(base.rglob("*.wav")):
        if wav.stem in found:
            raise RuntimeError(f"duplicate CirCor recording name {wav.stem}: {found[wav.stem]} and {wav}")
        found[wav.stem] = wav
    if not found:
        raise RuntimeError(f"no CirCor .wav files found under {base}")
    return found


def build_circor_index(root: Path, circor_dir: str, meta_csv: Path, policy: str,
                       min_duration: float, log=None) -> pd.DataFrame:
    meta = read_circor_metadata(meta_csv)
    wavs = find_circor_wavs(root, circor_dir)
    rows, skipped = [], {"no_metadata_for_patient": 0, "bad_filename": 0}
    for stem, wav in wavs.items():
        m = PATIENT_STEM.match(stem)
        if not m:
            skipped["bad_filename"] += 1
            continue
        pid, site = m.group("pid"), m.group("site").upper()
        if pid not in meta:
            skipped["no_metadata_for_patient"] += 1
            continue
        info = meta[pid]
        status, label = "ok", 0
        if info["murmur"] == "Unknown":
            status = "unknown"
        elif info["murmur"] == "Present":
            if not info["locs"]:
                status = "ambiguous_location"
            elif site in info["locs"]:
                label = 1
            elif policy == "exclude":
                status = "nonsite_excluded"
        dur, rms, clip = 0.0, 0.0, 0.0
        try:
            si = sf.info(str(wav))
            dur = si.frames / si.samplerate
            if dur < min_duration and status == "ok":
                status = "too_short"
        except Exception as e:
            status = "unreadable"
            if log:
                log.warning(f"unreadable {wav}: {e}")
        rows.append({"dataset": "circor", "path": _relpath(wav, root), "rec_id": stem, "patient_id": pid,
                     "site": site, "label": label, "class_name": ["Absent", "Present"][label],
                     "status": status, "duration": round(dur, 3), "rms": rms, "clip_frac": clip,
                     "patient_murmur": info["murmur"]})
    if log:
        log.info(f"CirCor: skipped {skipped}")
    if not rows:
        raise RuntimeError("no CirCor recordings matched the metadata")
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def expand_windows(index: pd.DataFrame, segment_seconds: float, hop_seconds: Optional[float]) -> pd.DataFrame:
    """One row per window. hop=None gives one centred window per recording (used for Yaseen)."""
    out = []
    for r in index.itertuples(index=False):
        dur = float(r.duration)
        if hop_seconds is None:
            starts = [max(0.0, (dur - segment_seconds) / 2.0)]
        elif dur <= segment_seconds:
            starts = [0.0]
        else:
            starts = list(np.arange(0.0, dur - segment_seconds + 1e-9, hop_seconds))
            last = dur - segment_seconds
            if starts[-1] < last - 0.5:          # keep the tail of the recording
                starts.append(last)
        for s in starts:
            d = r._asdict()
            d["start"] = round(float(s), 3)
            out.append(d)
    return pd.DataFrame(out)
