"""Synthetic dataset with the SAME folder layout as the real data. For testing code only: the sounds are not heart sounds.

  python scripts/make_synthetic.py --root data

data/heart_sound_dataset/{AS,MR,MS,MVP,N}/AS_New_3주기/*.wav   (8 kHz, a source/ folder is included on purpose)
data/circor/training_data.csv
data/circor/training_data/training_data/<pid>_<AV|PV|TV|MV>.{wav,tsv,hea} and <pid>.txt
data/circor/{LICENSE.txt,RECORDS,SHA256SUMS.txt}

The synthetic murmur is a 250 Hz tone in systole, on the listed auscultation sites only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SITES = ["AV", "PV", "TV", "MV"]


def heart_like(sr: float, dur: float, murmur: bool, rng) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for c in np.arange(0.0, dur, 0.8):
        for c0, f0 in ((c, 60.0), (c + 0.3, 50.0)):
            m = (t >= c0) & (t < c0 + 0.1)
            x[m] += np.sin(2 * np.pi * f0 * (t[m] - c0) * 2) * np.exp(-30 * (t[m] - c0))
    x += rng.normal(0, 0.05, len(t))
    if murmur:
        ph = t % 0.8
        x += 0.25 * np.sin(2 * np.pi * 250 * t) * ((ph > 0.12) & (ph < 0.33))
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def tsv(dur: float) -> str:
    rows = []
    for c in np.arange(0.0, dur, 0.8):
        for a, b, s in ((0.0, 0.1, 1), (0.1, 0.35, 2), (0.35, 0.45, 3), (0.45, 0.8, 4)):
            if c + b <= dur:
                rows.append(f"{c + a:.3f}\t{c + b:.3f}\t{s}")
    return "\n".join(rows) + "\n"


def main(root: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_dir = root / "heart_sound_dataset"
    for cls in ["AS", "MR", "MS", "MVP", "N"]:
        d = y_dir / cls / f"{cls}_New_3주기"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(14):
            sf.write(d / f"New_{cls}_{i + 1:03d}.wav", heart_like(8000, rng.uniform(3, 6), cls != "N", rng), 8000)
    (y_dir / "source").mkdir(exist_ok=True)
    (y_dir / "source" / "README.md").write_text("original rar archives live here; not part of the index\n")

    c_dir = root / "circor"
    w_dir = c_dir / "training_data" / "training_data"
    w_dir.mkdir(parents=True, exist_ok=True)
    status = ["Present"] * 9 + ["Absent"] * 16 + ["Unknown"] * 5
    rows = []
    for i, st in enumerate(status):
        pid = str(20000 + i)
        locs = "AV+MV" if st == "Present" else ("nan" if st == "Absent" else "AV+PV+TV+MV")
        if st == "Present" and i == 8:
            locs = "nan"                                     # one ambiguous-location patient
        rows.append({"Patient ID": pid, "Recording locations:": "AV+PV+TV+MV", "Age": "Child",
                     "Murmur": st, "Murmur locations": locs,
                     "Systolic murmur timing": "Holosystolic" if st == "Present" else "nan",
                     "Outcome": "Abnormal" if st == "Present" else "Normal"})
        for site in SITES:
            dur = float(rng.uniform(6, 10))
            murmur = st == "Present" and site in ("AV", "MV")
            sf.write(w_dir / f"{pid}_{site}.wav", heart_like(4000, dur, murmur, rng), 4000)
            (w_dir / f"{pid}_{site}.tsv").write_text(tsv(dur))
            (w_dir / f"{pid}_{site}.hea").write_text(f"{pid}_{site} 1 4000 {int(dur * 4000)}\n")
        (w_dir / f"{pid}.txt").write_text(f"{pid}\nMurmur: {st}\n")
    pd.DataFrame(rows).to_csv(c_dir / "training_data.csv", index=False)
    (c_dir / "LICENSE.txt").write_text("synthetic test data\n")
    (c_dir / "RECORDS").write_text("\n".join(f"{int(r['Patient ID'])}" for r in rows) + "\n")
    (c_dir / "SHA256SUMS.txt").write_text("")
    print(f"synthetic data written under {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(Path(a.root), a.seed)
