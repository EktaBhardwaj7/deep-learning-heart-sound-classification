"""
Verify and clean Yaseen + CirCor audio files.

Checks per file:
  - readable by soundfile
  - duration >= min duration (default 1.0 s)
  - not silent (RMS above threshold)
  - not clipped (fraction of samples at full-scale below threshold)

Usage:
  python data/clean_and_verify.py --root data/heart_sound_dataset
  python data/clean_and_verify.py --root data/circor --min-dur 3.0 --delete
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def check_file(path: Path, min_dur: float, silence_rms: float, clip_frac: float):
    """Return (ok: bool, reason: str, info: dict)."""
    try:
        info = sf.info(path)
        dur = info.duration
        sr = info.samplerate
        ch = info.channels
    except Exception as e:
        return False, f"unreadable: {e}", {}

    if dur < min_dur:
        return False, f"too short ({dur:.2f}s)", {"dur": dur, "sr": sr, "ch": ch}

    try:
        y, _ = sf.read(str(path), always_2d=False)
    except Exception as e:
        return False, f"read error: {e}", {"dur": dur, "sr": sr, "ch": ch}

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < silence_rms:
        return False, f"silent (rms={rms:.5f})", {"dur": dur, "sr": sr, "ch": ch, "rms": rms}

    peak = float(np.max(np.abs(y))) + 1e-12
    clip_ratio = float(np.mean(np.abs(y) >= 0.999))
    if clip_ratio > clip_frac:
        return False, f"clipped ({clip_ratio:.3f})", {"dur": dur, "sr": sr, "ch": ch, "clip": clip_ratio}

    return True, "ok", {"dur": dur, "sr": sr, "ch": ch, "rms": rms, "peak": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--silence-rms", type=float, default=1e-5)
    ap.add_argument("--clip-frac", type=float, default=0.05)
    ap.add_argument("--delete", action="store_true",
                    help="delete bad files (otherwise just report)")
    args = ap.parse_args()

    root = Path(args.root)
    wavs = sorted(root.rglob("*.wav"))
    print(f"Root: {root}")
    print(f"WAV files found: {len(wavs)}")

    bad = []
    good = 0
    durations = []

    for i, wav in enumerate(wavs):
        ok, reason, info = check_file(wav, args.min_dur, args.silence_rms, args.clip_frac)
        if ok:
            good += 1
            durations.append(info["dur"])
        else:
            bad.append((wav, reason))
        if (i + 1) % 500 == 0:
            print(f"  checked {i+1}/{len(wavs)}")

    print()
    print("=" * 60)
    print(f"GOOD: {good}")
    print(f"BAD : {len(bad)}")
    if durations:
        d = np.array(durations)
        print(f"Duration min / p50 / p95 / max : "
              f"{d.min():.2f} / {np.median(d):.2f} / {np.percentile(d, 95):.2f} / {d.max():.2f} s")
    print("=" * 60)

    if bad:
        print("\nBad files (first 20):")
        for w, r in bad[:20]:
            print(f"  {w.relative_to(root)}  ->  {r}")

        if args.delete:
            for w, _ in bad:
                try:
                    w.unlink()
                except Exception as e:
                    print(f"  failed to delete {w}: {e}")
            print(f"\nDeleted {len(bad)} bad files.")
            # Also remove any orphan .hea/.tsv for deleted wavs
            stems = {w.stem for w, _ in bad}
            for ext in (".hea", ".tsv"):
                for f in root.rglob(f"*{ext}"):
                    if f.stem in stems:
                        try:
                            f.unlink()
                        except Exception:
                            pass

    # Report per-class counts if this is Yaseen
    yaseen_dirs = ["AS", "MR", "MS", "MVP", "N"]
    if any((root / c).exists() for c in yaseen_dirs):
        print("\nPer-class counts (Yaseen layout):")
        for c in yaseen_dirs:
            d = root / c
            if d.exists():
                n = len(list(d.rglob("*.wav")))
                print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
