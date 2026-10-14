"""
Prepare Yaseen labels CSV matching the CirCor schema.

Produces: data/labels_yaseen.csv
Columns:  filename, segment_start, label, class, patient_id, location, use_for_training

Run once:
  python data/prepare_yaseen.py --root data/heart_sound_dataset --out data/labels_yaseen.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf


CLASS_TO_LABEL = {"AS": 0, "MR": 1, "MS": 2, "MVP": 3, "N": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment-sec", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.root)
    hop = args.segment_sec * (1.0 - args.overlap)
    rows = []

    for cls_name, label in CLASS_TO_LABEL.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} missing")
            continue
        for wav in sorted(cls_dir.rglob("*.wav")):
            info = sf.info(wav)
            dur = info.duration

            if dur <= args.segment_sec:
                starts = [0.0]
            else:
                n = int((dur - args.segment_sec) / hop) + 1
                starts = [round(s * hop, 3) for s in range(n)]
                last_start = dur - args.segment_sec
                if starts[-1] < last_start - 0.01:
                    starts.append(round(last_start, 3))

            for s in starts:
                rows.append({
                    "filename": str(wav.relative_to(root).as_posix()),
                    "segment_start": float(s),
                    "label": label,
                    "class": cls_name,
                    "patient_id": wav.stem,   # pseudo-ID; documented limitation
                    "location": "NA",
                    "use_for_training": 1,
                })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.groupby("class").size())


if __name__ == "__main__":
    main()