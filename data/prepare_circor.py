"""
Prepare CirCor labels CSV with LOCATION-AWARE murmur labels.

Key improvement over the previous version:
  The patient-level Murmur label is not applied to every recording.
  Instead, the Murmur locations column tells us which auscultation
  sites actually contain the murmur. A recording's label depends on
  whether its location is in the patient's murmur-location list.

Rules:
  Murmur = Absent  -> all four recordings get label 0
  Murmur = Unknown -> all four get label 2 (masked from training)
  Murmur = Present:
      If Murmur locations is populated:
          recordings at listed locations -> label 1
          other locations -> label 0
      If Murmur locations is empty (rare):
          fallback -> all locations get label 1 (documented in log)

Produces: data/labels_circor.csv
Columns: filename, segment_start, label, class, patient_id, location,
         murmur_locations, use_for_training

Run:
  python data/prepare_circor.py --root data/circor --out data/labels_circor.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf


def parse_filename(stem: str):
    """Return (patient_id, location_code). Filenames look like 49574_AV."""
    parts = stem.split("_")
    patient_id = parts[0]
    location = "NA"
    for p in parts[1:]:
        if p.upper() in {"AV", "PV", "TV", "MV"}:
            location = p.upper()
            break
    return patient_id, location


def parse_murmur_locations(value) -> set[str]:
    """Parse the Murmur locations column into a set of codes."""
    if pd.isna(value):
        return set()
    s = str(value)
    codes = set()
    for token in s.replace(";", ",").split(","):
        t = token.strip().upper()
        if t in {"AV", "PV", "TV", "MV"}:
            codes.add(t)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment-sec", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--min-seg-sec", type=float, default=3.0)
    ap.add_argument("--fallback-present-all", action="store_true", default=True,
                    help="when Present has no location list, label all sites 1")
    args = ap.parse_args()

    root = Path(args.root)
    meta_hits = list(root.rglob("training_data.csv"))
    if not meta_hits:
        raise SystemExit("training_data.csv not found")
    meta_path = meta_hits[0]
    meta = pd.read_csv(meta_path)
    print(f"Metadata: {meta_path}  shape={meta.shape}")

    # Build patient -> (overall_status, murmur_locations, label_map)
    pat_status = {}
    pat_locs = {}
    for _, r in meta.iterrows():
        pid = str(r["Patient ID"])
        status = str(r["Murmur"]).strip().lower()
        locs = parse_murmur_locations(r["Murmur locations"])
        pat_status[pid] = status
        pat_locs[pid] = locs

    status_counts = Counter(pat_status.values())
    print(f"Patient status counts: {dict(status_counts)}")

    present_no_locs = sum(1 for p, s in pat_status.items()
                          if s == "present" and not pat_locs[p])
    print(f"Present patients with empty Murmur locations: {present_no_locs}")

    # Counters for reporting
    loc_label_counts = Counter()
    hop = args.segment_sec * (1.0 - args.overlap)

    wavs = sorted(root.rglob("*.wav"))
    print(f"Found {len(wavs)} wav files")

    rows = []
    for wav in wavs:
        pid, loc = parse_filename(wav.stem)
        status = pat_status.get(pid)
        if status is None:
            continue
        murmur_locs = pat_locs.get(pid, set())

        # Determine this recording's label
        if status == "absent":
            label, cls, use = 0, "Absent", 1
        elif status == "unknown":
            label, cls, use = 2, "Unknown", 0
        elif status == "present":
            if murmur_locs:
                if loc in murmur_locs:
                    label, cls, use = 1, "Present", 1
                else:
                    label, cls, use = 0, "Absent", 1
            else:
                # Present but no location info -> fallback
                if args.fallback_present_all:
                    label, cls, use = 1, "Present", 1
                else:
                    label, cls, use = 0, "Absent", 1
        else:
            continue

        loc_label_counts[(loc, cls)] += 1

        try:
            info = sf.info(wav)
        except Exception:
            continue
        dur = info.duration
        if dur < args.min_seg_sec:
            continue

        if dur <= args.segment_sec:
            starts = [0.0]
        else:
            n = int((dur - args.segment_sec) / hop) + 1
            starts = [round(s * hop, 3) for s in range(n)]
            last = dur - args.segment_sec
            if starts[-1] < last - 0.01:
                starts.append(round(last, 3))

        rel = wav.relative_to(root).as_posix()
        murmur_locs_str = ",".join(sorted(murmur_locs)) if murmur_locs else ""
        for s in starts:
            rows.append({
                "filename": rel,
                "segment_start": float(s),
                "label": label,
                "class": cls,
                "patient_id": pid,
                "location": loc,
                "murmur_locations": murmur_locs_str,
                "use_for_training": use,
            })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} segment rows to {args.out}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print("\nSegment-level class distribution:")
    print(df.groupby(["class", "use_for_training"]).size())
    print("\nRecording-level location vs label:")
    loc_df = pd.DataFrame(
        [(loc, cls, cnt) for (loc, cls), cnt in loc_label_counts.items()],
        columns=["location", "class", "count"],
    )
    print(loc_df.pivot_table(index="location", columns="class",
                             values="count", fill_value=0))


if __name__ == "__main__":
    main()
