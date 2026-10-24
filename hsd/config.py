"""Configuration: YAML files with `extends:` inheritance and dotted command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

CONFIG_DIR = Path("configs")


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _load_file(path.parent / parent)
        data = deep_merge(base, data)
    return data


def set_dotted(cfg: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> dict:
    path = Path(path)
    if not path.exists():
        alt = CONFIG_DIR / path.name
        path = alt if alt.exists() else path
    cfg = _load_file(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, val = item.split("=", 1)
        set_dotted(cfg, key, val)
    validate(cfg)
    cfg["_meta"] = {"config_path": str(path)}
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    need = {"project", "data", "features", "model", "train", "eval"}
    missing = need - set(cfg)
    if missing:
        raise ValueError(f"config missing sections: {sorted(missing)}")
    d, f, m, t = cfg["data"], cfg["features"], cfg["model"], cfg["train"]
    if f["representation"] not in ("logmel", "raw"):
        raise ValueError("features.representation must be logmel or raw")
    if m["name"] not in ("cnn1d", "resnet1d", "cnn2d", "resnet2d", "crnn", "bilstm", "transformer"):
        raise ValueError(f"unknown model.name {m['name']}")
    if m.get("pooling", "mean") not in ("mean", "attention"):
        raise ValueError("model.pooling must be mean or attention")
    if f["representation"] == "raw" and m["name"] not in ("cnn1d", "resnet1d"):
        raise ValueError("raw representation needs a 1-D model (cnn1d or resnet1d)")
    if f["representation"] == "logmel" and m["name"] in ("cnn1d", "resnet1d"):
        raise ValueError("log-mel representation needs a 2-D or sequence model")
    if not set(d["tasks"]) <= {"yaseen", "circor"} or not d["tasks"]:
        raise ValueError("data.tasks must be a non-empty subset of [yaseen, circor]")
    if t["optimizer"]["name"] not in ("adam", "adamw", "sgd"):
        raise ValueError("train.optimizer.name must be adam, adamw or sgd")
    if d["nonmurmur_site_policy"] not in ("absent", "exclude"):
        raise ValueError("data.nonmurmur_site_policy must be absent or exclude")
    if cfg["eval"]["aggregation"] not in ("auto", "mean", "max", "median", "noisy_or", "vote"):
        raise ValueError("eval.aggregation invalid")


def run_paths(cfg: dict) -> dict:
    name = cfg["project"]["name"]
    run = Path(cfg["project"]["runs_dir"]) / name
    ckpt = Path(cfg["project"]["checkpoint_dir"]) / name
    return {"run": run, "ckpt": ckpt, "eval": run / "eval", "explain": run / "explain",
            "robust": run / "robustness", "uncert": run / "uncertainty", "export": run / "export"}


def save_resolved(cfg: dict, path: Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(clean, sort_keys=False, allow_unicode=True), encoding="utf-8")
