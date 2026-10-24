"""Check your data folders BEFORE training. Prints what was found and what is missing.

  python scripts/check_data.py --config configs/joint_main.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsd.config import load_config  # noqa: E402
from hsd.labels import build_circor_index, build_yaseen_index, read_circor_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/joint_main.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg["data"]["root"])
    ok = True
    print(f"data root: {root.resolve()}")

    y_dir = root / cfg["data"]["yaseen_dir"]
    for cls in cfg["data"]["class_names"]["yaseen"]:
        n = len(list((y_dir / cls).rglob("*.wav"))) if (y_dir / cls).exists() else 0
        flag = "OK " if n else "MISSING"
        ok &= bool(n)
        print(f"  [{flag}] Yaseen {cls}: {n} wav files")

    c_dir = root / cfg["data"]["circor_dir"]
    meta = root / cfg["data"]["circor_meta"]
    wavs = list(c_dir.rglob("*.wav")) if c_dir.exists() else []
    print(f"  [{'OK ' if wavs else 'MISSING'}] CirCor: {len(wavs)} wav files under {c_dir}")
    print(f"  [{'OK ' if meta.exists() else 'MISSING'}] CirCor metadata: {meta}")
    ok &= bool(wavs) and meta.exists()
    if meta.exists():
        m = read_circor_metadata(meta)
        counts = {}
        for v in m.values():
            counts[v["murmur"]] = counts.get(v["murmur"], 0) + 1
        print(f"  patients in metadata: {len(m)}  murmur status: {counts}")
        idx = build_circor_index(root, cfg["data"]["circor_dir"], meta, cfg["data"]["nonmurmur_site_policy"], 0.5)
        print("  CirCor recording status:\n" + idx.groupby("status").size().to_string())
    if y_dir.exists():
        yi = build_yaseen_index(root, cfg["data"]["yaseen_dir"], cfg["data"]["class_names"]["yaseen"], 0.5)
        print("  Yaseen recording status:\n" + yi.groupby("status").size().to_string())
    print("\nRESULT:", "READY" if ok else "NOT READY (fix the MISSING lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
