"""Leakage-free splits. Yaseen is split by recording; CirCor by patient (all recordings of a patient stay together).

Patients whose murmur status is 'Unknown' are held out entirely (split = 'unknown') and used only for the
uncertainty analysis. Patients labelled Present with no listed murmur site are excluded (split = 'excluded').
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _group_split(groups: pd.DataFrame, strat: str, train: float, val: float, seed: int):
    """groups: one row per group with columns [key, strat]. Returns three key sets."""
    def split(df, test_size):
        try:
            return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[strat])
        except ValueError:                       # a stratum has a single member
            return train_test_split(df, test_size=test_size, random_state=seed)
    tr, tmp = split(groups, 1.0 - train)
    va, te = split(tmp, 1.0 - val / (1.0 - train))
    return set(tr["key"]), set(va["key"]), set(te["key"])


def assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = cfg["project"]["seed"]
    idx = index.copy()
    idx["split"] = "excluded"
    tasks = cfg["data"]["tasks"]

    if "yaseen" in tasks:
        y = idx[idx.dataset == "yaseen"]
        ok = y[y.status == "ok"]
        groups = ok.drop_duplicates("rec_id")[["rec_id", "class_name"]].rename(columns={"rec_id": "key", "class_name": "strat"})
        tr, va, te = _group_split(groups, "strat", *cfg["data"]["yaseen_split"], seed)
        mapping = {**{k: "train" for k in tr}, **{k: "val" for k in va}, **{k: "test" for k in te}}
        m = idx.dataset.eq("yaseen")
        idx.loc[m, "split"] = idx.loc[m, "rec_id"].map(mapping).fillna("excluded")

    if "circor" in tasks:
        c = idx[idx.dataset == "circor"]
        patients = c.groupby("patient_id").agg(
            murmur=("patient_murmur", "first"),
            has_unknown=("status", lambda s: (s == "unknown").any()),
            ambiguous=("status", lambda s: (s == "ambiguous_location").any()),
        ).reset_index()
        held = patients[patients.has_unknown]
        excl = patients[(~patients.has_unknown) & patients.ambiguous]
        eligible = patients[(~patients.has_unknown) & (~patients.ambiguous)]
        groups = eligible[["patient_id", "murmur"]].rename(columns={"patient_id": "key", "murmur": "strat"})
        tr, va, te = _group_split(groups, "strat", *cfg["data"]["circor_split"], seed)
        mapping = {**{k: "train" for k in tr}, **{k: "val" for k in va}, **{k: "test" for k in te},
                   **{k: "unknown" for k in held.patient_id}, **{k: "excluded" for k in excl.patient_id}}
        m = idx.dataset.eq("circor")
        idx.loc[m, "split"] = idx.loc[m, "patient_id"].map(mapping).fillna("excluded")
    return idx


def check_no_leakage(idx: pd.DataFrame) -> None:
    for ds, key in (("yaseen", "rec_id"), ("circor", "patient_id")):
        d = idx[(idx.dataset == ds) & idx.split.isin(["train", "val", "test"])]
        sets = {s: set(d[d.split == s][key]) for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(f"leakage in {ds}: {len(overlap)} {key} shared by {a} and {b}")


def _fingerprint(idx: pd.DataFrame, cfg: dict) -> str:
    payload = json.dumps({"seed": cfg["project"]["seed"], "tasks": cfg["data"]["tasks"],
                          "ys": cfg["data"]["yaseen_split"], "cs": cfg["data"]["circor_split"],
                          "policy": cfg["data"]["nonmurmur_site_policy"],
                          "paths": sorted(idx.path.tolist())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_or_create_splits(index: pd.DataFrame, cfg: dict, out_dir: Path, log=None) -> pd.DataFrame:
    """Splits are created once and then reused, so every run and every model sees the same partition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, meta = out_dir / "index.csv", out_dir / "splits.json"
    fp = _fingerprint(index, cfg)
    if csv.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get("fingerprint") == fp:
            if log:
                log.info(f"reusing splits from {csv} (fingerprint {fp})")
            return pd.read_csv(csv, dtype={"rec_id": str, "patient_id": str})
    idx = assign_splits(index, cfg)
    check_no_leakage(idx)
    idx.to_csv(csv, index=False)
    summary = {"fingerprint": fp,
               "counts": idx.groupby(["dataset", "split", "status"]).size().reset_index(name="n").to_dict("records")}
    meta.write_text(json.dumps(summary, indent=2, default=str))
    if log:
        log.info("splits created:\n" + idx.groupby(["dataset", "split"]).size().to_string())
    return idx
