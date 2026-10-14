"""
Entry point: trains, evaluates, or runs multi-task mix experiments.

Examples
--------
# Yaseen single-dataset
python main.py train    --config configs/config.yaml
python main.py evaluate --config configs/config.yaml --checkpoint checkpoints/best.pth

# Multi-dataset mix
python main.py train_mix    --config configs/config_mix.yaml
python main.py evaluate_mix --config configs/config_mix.yaml --checkpoint checkpoints/mix/best.pth

# Resume
python main.py train --config configs/config.yaml --resume checkpoints/last.pth
"""
from __future__ import annotations

# --- Fix Python 3.14 forkserver default for PyTorch DataLoader workers ---
import multiprocessing as _mp
try:
    _mp.set_start_method('fork', force=True)
except RuntimeError:
    pass
# -------------------------------------------------------------------------

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("heart-sound")
    sub = p.add_subparsers(dest="cmd", required=True)

    for cmd in ("train", "evaluate", "train_mix", "evaluate_mix"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--config", type=str, required=True)
        sp.add_argument("--resume", type=str, default=None,
                        help="(train*) resume from checkpoint")
        sp.add_argument("--checkpoint", type=str, default=None,
                        help="(evaluate*) checkpoint to load")
        sp.add_argument("--set", nargs="*", default=[],
                        help="config overrides, e.g. --set training.epochs=10")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _apply_overrides(cfg, overrides):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override: {item}")
        key, val = item.split("=", 1)
        node = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        node[keys[-1]] = val
    return cfg


def load_config(path, overrides=None):
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _load_labels(cfg):
    df = pd.read_csv(cfg["data"]["labels_csv"])
    need = {"filename", "label", "class"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing columns: {missing}")
    return df.sort_values("filename").reset_index(drop=True)


def _logmel_cfg(cfg):
    f = cfg["features"]
    return dict(
        n_mels=f["n_mels"], n_fft=f["n_fft"],
        hop=f["hop_length"], win=f["win_length"],
        fmin=f["fmin"], fmax=f["fmax"],
    )


def _make_loaders(cfg, labels_df, logger):
    from utils import make_dataloaders
    return make_dataloaders(
        labels_df=labels_df,
        dataset_root=Path(cfg["data"]["dataset_root"]),
        representation=cfg["features"]["representation"],
        sr=cfg["data"]["sample_rate"],
        duration=cfg["data"]["duration"],
        n_mels=cfg["features"]["n_mels"],
        batch_size=cfg["training"]["batch_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["project"]["seed"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        normalize=cfg["features"]["normalize"],
        apply_bandpass=cfg["features"]["apply_bandpass"],
        cache_dir=Path("features/cache"),
        logmel_cfg=_logmel_cfg(cfg),
        logger=logger,
    )


def _build_model(cfg):
    from models import build_model
    return build_model(
        name=cfg["model"]["name"],
        num_classes=cfg["data"]["num_classes"],
        n_mels=cfg["features"]["n_mels"],
        dropout=cfg["model"]["dropout"],
        base_channels=cfg["model"]["base_channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        nhead=cfg["model"]["nhead"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
    )


# ---------------------------------------------------------------------------
# Single-dataset train
# ---------------------------------------------------------------------------
def cmd_train(args):
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["project"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["checkpoint"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    from utils import get_logger, TensorBoardWriter, plot_training_curves
    logger = get_logger("heart_sound", log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])
    logger.info(f"Command: train | config={args.config}")

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    labels_df = _load_labels(cfg)
    logger.info(f"Loaded {len(labels_df)} labelled recordings")
    logger.info(f"Class counts:\n{labels_df['class'].value_counts().to_string()}")

    train_loader, val_loader, test_loader, normalizer, info = _make_loaders(cfg, labels_df, logger)
    logger.info(f"Splits -> train={info['n_train']} val={info['n_val']} test={info['n_test']}")
    logger.info(f"Input shape per sample: {info['input_shape']} | target_frames={info['target_frames']}")

    (out_dir / "normalizer.json").write_text(json.dumps(normalizer.state_dict(), indent=2))

    model = _build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['model']['name']} | trainable params: {n_params:,}")

    from training import Trainer, build_optimizer, build_scheduler, build_loss
    optimizer = build_optimizer(model, cfg["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["training"]["scheduler"])
    loss_fn = build_loss(cfg["training"]["loss"])

    tb = TensorBoardWriter(out_dir / "tb") if cfg["logging"].get("tensorboard") else None
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                      device, cfg["training"], logger, tb_writer=tb,
                      checkpoint_dir=ckpt_dir)

    resume = args.resume or cfg["checkpoint"].get("resume")
    if resume:
        trainer.resume(resume)

    history = trainer.fit()
    plot_training_curves(history, out_dir / "training_curves.png")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    if tb is not None:
        tb.close()
    logger.info(f"Training complete. Artifacts in {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Single-dataset evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args):
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["project"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    from utils import (get_logger, compute_metrics, per_class_report,
                       confusion, plot_confusion_matrix)
    logger = get_logger("heart_sound", log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])
    logger.info(f"Command: evaluate | checkpoint={args.checkpoint}")

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = _load_labels(cfg)
    _, _, test_loader, _, _ = _make_loaders(cfg, labels_df, logger)

    model = _build_model(cfg).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info(f"Loaded checkpoint {ckpt} (epoch {state.get('epoch')})")

    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits.float(), dim=-1)
            all_pred.append(probs.argmax(dim=-1).cpu().numpy())
            all_true.append(y.numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    names = cfg["data"]["class_names"]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = per_class_report(y_true, y_pred, names)
    cm = confusion(y_true, y_pred, num_classes=cfg["data"]["num_classes"])

    logger.info("Test metrics: " + json.dumps(metrics, indent=2))
    print("\nPer-class report:\n" + report)

    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.txt").write_text(report)
    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, names, out_dir / "confusion_matrix.png", normalize=True)
    plot_confusion_matrix(cm, names, out_dir / "confusion_matrix_raw.png", normalize=False)
    logger.info(f"Evaluation artifacts written to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Mix train
# ---------------------------------------------------------------------------
def cmd_train_mix(args):
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["project"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["checkpoint"]["dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    from utils import get_logger, TensorBoardWriter
    from utils.multi_dataset import build_loaders
    from models.multi_head import MultiTaskModel
    from training.trainer_mix import MultiTaskTrainer
    from training.trainer import build_optimizer, build_scheduler

    logger = get_logger("heart_sound_mix",
                        log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])
    logger.info("Command: train_mix")

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    loaders, info = build_loaders(
        yaseen_root=Path(cfg["data"]["yaseen_root"]),
        circor_root=Path(cfg["data"]["circor_root"]),
        yaseen_csv=Path(cfg["data"]["yaseen_csv"]),
        circor_csv=Path(cfg["data"]["circor_csv"]),
        batch_yaseen=cfg["training"]["batch_yaseen"],
        batch_circor=cfg["training"]["batch_circor"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["project"]["seed"],
        use_specaugment=cfg["training"].get("use_specaugment", True),
    )
    logger.info(f"Yaseen splits: {info['yaseen_counts']}")
    logger.info(f"CirCor splits: {info['circor_counts']}")
    logger.info(f"CirCor unknowns held out: val={info['circor_unknown_in_val']} test={info['circor_unknown_in_test']}")
    logger.info(f"Input shape: {info['input_shape']}")

    model = MultiTaskModel(
        num_yaseen=cfg["data"]["num_yaseen_classes"],
        num_circor=cfg["data"]["num_circor_classes"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MultiTaskModel trainable params: {n_params:,}")

    optimizer = build_optimizer(model, cfg["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["training"]["scheduler"])

    tb = TensorBoardWriter(out_dir / "tb") if cfg["logging"].get("tensorboard") else None
    trainer = MultiTaskTrainer(model, loaders, optimizer, scheduler,
                               device, cfg["training"], logger,
                               tb_writer=tb, checkpoint_dir=ckpt_dir,
                               yaseen_class_weights=info.get("yaseen_class_weights"),
                               circor_class_weights=info.get("circor_class_weights"))
    resume = args.resume or cfg["checkpoint"].get("resume")
    if resume:
        trainer.resume(resume)

    trainer.fit()
    (out_dir / "history.json").write_text(json.dumps(trainer.history, indent=2))
    if tb is not None:
        tb.close()
    logger.info(f"Mix training done. Checkpoints in {ckpt_dir.resolve()}")


# ---------------------------------------------------------------------------
# Mix evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate_mix(args):
    """Evaluate mix model with segment-level AND recording-level metrics."""
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg["project"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    from utils import (get_logger, compute_metrics, per_class_report,
                       confusion, plot_confusion_matrix)
    from utils.multi_dataset import build_loaders
    from models.multi_head import MultiTaskModel
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, f1_score

    logger = get_logger("heart_sound_mix",
                        log_file=cfg["logging"]["log_file"],
                        level=cfg["logging"]["level"])
    logger.info(f"Command: evaluate_mix | ckpt={args.checkpoint}")

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, _ = build_loaders(
        yaseen_root=Path(cfg["data"]["yaseen_root"]),
        circor_root=Path(cfg["data"]["circor_root"]),
        yaseen_csv=Path(cfg["data"]["yaseen_csv"]),
        circor_csv=Path(cfg["data"]["circor_csv"]),
        batch_yaseen=cfg["training"]["batch_yaseen"],
        batch_circor=cfg["training"]["batch_circor"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["project"]["seed"],
        use_specaugment=False,
    )

    model = MultiTaskModel(
        num_yaseen=cfg["data"]["num_yaseen_classes"],
        num_circor=cfg["data"]["num_circor_classes"],
        base=cfg["model"]["base_channels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    def run_segment_level(task, loader):
        """Return DataFrame with columns: filename, label, prob_0..prob_K-1."""
        probs_all, labels_all, fnames = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                logits = model(x, task=task)
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                probs_all.append(p)
                labels_all.append(y.numpy())
                base = i * loader.batch_size
                for j in range(x.size(0)):
                    idx = base + j
                    if idx >= len(loader.dataset):
                        break
                    fnames.append(loader.dataset.labels.iloc[idx]["filename"])
        probs = np.concatenate(probs_all)
        labels = np.concatenate(labels_all)
        df = pd.DataFrame({"filename": fnames, "label": labels})
        for k in range(probs.shape[1]):
            df[f"prob_{k}"] = probs[:, k]
        return df

    def aggregate_recording(df, strategy="noisy_or"):
        """Group by filename. Multi-class -> mean pooling. Binary + noisy_or -> 1 - prod(1-p)."""
        n_classes = sum(1 for c in df.columns if c.startswith("prob_"))
        grouped = df.groupby("filename")
        rows = []
        for fname, g in grouped:
            label = int(g["label"].iloc[0])
            probs = g[[f"prob_{k}" for k in range(n_classes)]].values
            if n_classes == 2 and strategy == "noisy_or":
                p_present = 1.0 - np.prod(1.0 - probs[:, 1])
                p_absent = 1.0 - p_present
                agg = np.array([p_absent, p_present])
            elif strategy == "max":
                agg = probs.max(axis=0)
            else:
                agg = probs.mean(axis=0)
            agg = agg / (agg.sum() + 1e-12)
            rows.append({"filename": fname, "label": label,
                         **{f"prob_{k}": agg[k] for k in range(n_classes)}})
        return pd.DataFrame(rows)

    def find_best_threshold(y_true, y_prob_pos, min_sens=0.90):
        """Return threshold that maximizes precision subject to recall >= min_sens."""
        prec, rec, thr = precision_recall_curve(y_true, y_prob_pos)
        valid = np.where(rec >= min_sens)[0]
        if len(valid) == 0:
            return 0.5, float("nan"), float("nan")
        best_idx = valid[np.argmax(prec[valid])]
        if best_idx >= len(thr):
            return 0.5, float("nan"), float("nan")
        return float(thr[best_idx]), float(prec[best_idx]), float(rec[best_idx])

    for task, split_key, names in [
        ("yaseen", "yaseen_test", cfg["data"]["class_names_yaseen"]),
        ("circor", "circor_test", cfg["data"]["class_names_circor"]),
    ]:
        seg_df = run_segment_level(task, loaders[split_key])
        rec_df = aggregate_recording(seg_df, strategy="noisy_or" if task == "circor" else "mean")

        n_classes = len(names)

        # ---- Segment-level metrics
        seg_pred = seg_df[[f"prob_{k}" for k in range(n_classes)]].values.argmax(axis=1)
        seg_true = seg_df["label"].values
        seg_probs = seg_df[[f"prob_{k}" for k in range(n_classes)]].values
        seg_m = compute_metrics(seg_true, seg_pred, seg_probs)

        # ---- Recording-level metrics (default threshold)
        rec_pred = rec_df[[f"prob_{k}" for k in range(n_classes)]].values.argmax(axis=1)
        rec_true = rec_df["label"].values
        rec_probs = rec_df[[f"prob_{k}" for k in range(n_classes)]].values
        rec_m = compute_metrics(rec_true, rec_pred, rec_probs)

        logger.info(f"[{task}] SEGMENT-level: {json.dumps(seg_m, indent=2)}")
        logger.info(f"[{task}] RECORDING-level: {json.dumps(rec_m, indent=2)}")
        print(f"\n[{task}] SEGMENT metrics:")
        print(json.dumps(seg_m, indent=2))
        print(f"\n[{task}] RECORDING metrics:")
        print(json.dumps(rec_m, indent=2))

        # Save reports
        (out_dir / f"{task}_segment_metrics.json").write_text(json.dumps(seg_m, indent=2))
        (out_dir / f"{task}_recording_metrics.json").write_text(json.dumps(rec_m, indent=2))
        (out_dir / f"{task}_recording_report.txt").write_text(
            per_class_report(rec_true, rec_pred, names))

        plot_confusion_matrix(confusion(rec_true, rec_pred),
                              names, out_dir / f"{task}_recording_confusion.png", normalize=True)

        # ---- Threshold tuning (CirCor only)
        if task == "circor" and n_classes == 2:
            p_present = rec_df["prob_1"].values
            thr, prec_at, rec_at = find_best_threshold(rec_true, p_present, min_sens=0.90)
            pred_thr = (p_present >= thr).astype(int)
            f1_thr = f1_score(rec_true, pred_thr, average="macro", zero_division=0)
            logger.info(f"[circor] threshold for 90% sens: {thr:.3f} "
                        f"(prec={prec_at:.3f} rec={rec_at:.3f}) macro-F1={f1_thr:.4f}")
            print(f"\n[circor] Threshold for >=90% sensitivity: {thr:.3f}")
            print(f"[circor] At this threshold: macro-F1={f1_thr:.4f}")
            (out_dir / "circor_threshold.json").write_text(json.dumps({
                "threshold_for_90pct_sens": thr,
                "precision_at_threshold": prec_at,
                "recall_at_threshold": rec_at,
                "macro_f1_at_threshold": f1_thr,
            }, indent=2))



# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "evaluate":
        if not args.checkpoint:
            print("--checkpoint required for evaluate", file=sys.stderr); sys.exit(1)
        cmd_evaluate(args)
    elif args.cmd == "train_mix":
        cmd_train_mix(args)
    elif args.cmd == "evaluate_mix":
        if not args.checkpoint:
            print("--checkpoint required for evaluate_mix", file=sys.stderr); sys.exit(1)
        cmd_evaluate_mix(args)


if __name__ == "__main__":
    main()