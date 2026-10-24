"""Command line.

  python -m hsd params     --config configs/rq1_logmel.yaml
  python -m hsd prepare    --config configs/joint_main.yaml
  python -m hsd train      --config configs/joint_main.yaml [--resume auto|PATH]
  python -m hsd evaluate   --config configs/joint_main.yaml
  python -m hsd robustness | uncertainty | explain | export | device-eval | report  --config ...
  python -m hsd infer      --model runs/joint_main/export/model_int8.onnx --meta runs/joint_main/export/meta.json \
                           --task circor --wav some.wav
  python -m hsd all        --config configs/joint_main.yaml

Any setting can be changed on the command line: --set train.epochs=5 model.name=crnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import load_config, run_paths, save_resolved
from .utils import get_logger, pick_device, set_seed, write_json

COMMANDS = ["params", "prepare", "train", "evaluate", "robustness", "uncertainty", "explain", "export",
            "device-eval", "infer", "report", "all"]


def _parser():
    p = argparse.ArgumentParser(prog="hsd", description="heart_screen: phonocardiogram screening pipeline")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in COMMANDS:
        sp = sub.add_parser(c)
        sp.add_argument("--config", default=None)
        sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
        sp.add_argument("--resume", default=None, help="train: 'auto' or a checkpoint path")
        sp.add_argument("--checkpoint", default=None, help="evaluation commands: weights to load (default best.pth)")
        sp.add_argument("--quiet", action="store_true")
        if c == "infer":
            sp.add_argument("--model", required=True)
            sp.add_argument("--meta", required=True)
            sp.add_argument("--task", required=True)
            sp.add_argument("--wav", required=True)
            sp.add_argument("--rule", default=None)
    return p


class Ctx:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config, args.set)
        self.paths = run_paths(self.cfg)
        self.paths["run"].mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"hsd.{self.cfg['project']['name']}", self.paths["run"] / "run.log",
                              level="WARNING" if args.quiet else "INFO")
        set_seed(int(self.cfg["project"]["seed"]))
        self.device = pick_device(self.cfg["project"].get("device", "auto"))
        save_resolved(self.cfg, self.paths["run"] / "config_resolved.yaml")
        self.log.info(f"command={args.cmd} config={self.cfg['_meta']['config_path']} device={self.device} "
                      f"hsd={__version__}")
        self._data = None

    @property
    def spec(self):
        from .audio import spec_from_cfg
        return spec_from_cfg(self.cfg)

    def data(self):
        """Index, splits, windows and loaders. Built once per process."""
        if self._data is None:
            from .pipeline import build_loaders, build_windows, prepare_index
            idx = prepare_index(self.cfg, self.log)
            windows = build_windows(self.cfg, idx, self.spec)
            loaders = build_loaders(self.cfg, windows, self.spec)
            for k, v in windows.items():
                self.log.info(f"windows {k}: {len(v)}")
            self._data = {"idx": idx, "windows": windows, "loaders": loaders}
        return self._data

    def model(self, checkpoint: str | None = None, use_ema: bool = True):
        from .models import build_model
        from .utils import load_checkpoint
        ckpt = Path(checkpoint) if checkpoint else self.paths["ckpt"] / "best.pth"
        if not ckpt.exists():
            ckpt = self.paths["ckpt"] / "last.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"no checkpoint in {self.paths['ckpt']}; run `train` first")
        ck = load_checkpoint(ckpt, self.device)
        tasks = ck["task_classes"]
        net = build_model(self.cfg, tasks).to(self.device)
        state = ck["ema"]["module"] if (use_ema and ck.get("ema")) else ck["model"]
        net.load_state_dict(state)
        net.eval()
        self.log.info(f"loaded {ckpt} (epoch {ck['epoch']}, {'EMA' if use_ema and ck.get('ema') else 'raw'} weights)")
        return net, ck


def cmd_params(ctx):
    from .models import build_model, count_params
    tasks = {t: len(ctx.cfg["data"]["class_names"][t]) for t in ctx.cfg["data"]["tasks"]}
    net = build_model(ctx.cfg, tasks)
    ctx.log.info(f"model {ctx.cfg['model']['name']} base={ctx.cfg['model']['base_channels']}: "
                 f"backbone={count_params(net.backbone):,} total={count_params(net):,}")
    print(json.dumps({"backbone_params": count_params(net.backbone), "total_params": count_params(net)}))


def cmd_prepare(ctx):
    d = ctx.data()
    ctx.log.info("prepare complete")
    write_json(ctx.paths["run"] / "windows_summary.json",
               {k: len(v) for k, v in d["windows"].items()})


def cmd_train(ctx):
    from .models import build_model, count_params
    from .pipeline import class_weights
    from .plots import training_curves
    from .trainer import Trainer
    from .utils import get_logger
    d = ctx.data()
    cw = class_weights(ctx.cfg, d["windows"])
    tasks = {t: len(ctx.cfg["data"]["class_names"][t]) for t in ctx.cfg["data"]["tasks"]}
    net = build_model(ctx.cfg, tasks)
    ctx.log.info(f"model {ctx.cfg['model']['name']}: backbone={count_params(net.backbone):,} params")
    ctx.log.info(f"class weights: {cw}")
    tr = Trainer(net, d["loaders"], ctx.cfg, ctx.device, ctx.log, ctx.paths["ckpt"], ctx.paths["run"], cw)
    if ctx.args.resume:
        target = ctx.paths["ckpt"] / "last.pth" if ctx.args.resume == "auto" else Path(ctx.args.resume)
        if target.exists():
            tr.resume(target)
        else:
            ctx.log.warning(f"no checkpoint at {target}; starting a new run")
    hist = tr.fit()
    training_curves(hist, ctx.paths["run"] / "training_curves.png", ctx.cfg["data"]["tasks"])
    write_json(ctx.paths["run"] / "model_info.json", {
        "tasks": ctx.cfg["data"]["tasks"], "task_classes": tasks, "spec": _spec_dict(ctx),
        "best_score": tr.best, "epochs_completed": len(hist)})
    ctx.log.info(f"training finished: best composite score {tr.best:.4f}")


def _spec_dict(ctx) -> dict:
    s = ctx.spec
    return {"sr": s.sr, "segment_seconds": s.segment_seconds, "representation": s.representation,
            "n_mels": s.n_mels, "n_fft": s.n_fft, "hop": s.hop, "win": s.win, "fmin": s.fmin, "fmax": s.fmax,
            "bandpass": list(s.bandpass) if s.bandpass else None}


def cmd_evaluate(ctx):
    from .evaluate import evaluate_task
    d = ctx.data()
    net, ck = ctx.model(ctx.args.checkpoint)
    amp = bool(ctx.cfg["train"].get("amp", True)) and ctx.device.type == "cuda"
    res = {}
    for task in ctx.cfg["data"]["tasks"]:
        res[task] = evaluate_task(task, net, d["loaders"], d["windows"], ctx.cfg, ctx.device,
                                  ctx.paths["eval"], ctx.log, amp)
    write_json(ctx.paths["eval"] / "summary.json", {"checkpoint_epoch": ck["epoch"], "tasks": res})
    ctx.log.info(f"evaluation written to {ctx.paths['eval']}")


def cmd_robustness(ctx):
    from .robustness import run_robustness
    d = ctx.data()
    net, _ = ctx.model(ctx.args.checkpoint)
    amp = bool(ctx.cfg["train"].get("amp", True)) and ctx.device.type == "cuda"
    run_robustness(ctx.cfg, net, d["windows"], ctx.spec, ctx.device, ctx.paths["robust"], ctx.log, amp)


def cmd_uncertainty(ctx):
    from .uncertainty import run_uncertainty
    d = ctx.data()
    net, _ = ctx.model(ctx.args.checkpoint)
    amp = bool(ctx.cfg["train"].get("amp", True)) and ctx.device.type == "cuda"
    run_uncertainty(ctx.cfg, net, {t: d["loaders"][t] for t in ctx.cfg["data"]["tasks"]}, d["windows"],
                    ctx.spec, ctx.device, ctx.paths["uncert"], ctx.log, amp)


def cmd_explain(ctx):
    from .explain import run_explain
    d = ctx.data()
    net, _ = ctx.model(ctx.args.checkpoint)
    run_explain(ctx.cfg, net, d["windows"], ctx.spec, ctx.device, ctx.paths["explain"], ctx.log)


def cmd_export(ctx):
    from .export import export_onnx, parity_report, quantize_int8
    from .evaluate import get_aggregation
    d = ctx.data()
    net, _ = ctx.model(ctx.args.checkpoint)
    out = ctx.paths["export"]
    out.mkdir(parents=True, exist_ok=True)
    fp32 = export_onnx(net, ctx.spec, out / "model_fp32.onnx")
    tasks = ctx.cfg["data"]["tasks"]
    sample = torch.from_numpy(_sample_batch(d["windows"], ctx, tasks[0]))
    parity = parity_report(net, fp32, sample, tasks)
    ctx.log.info(f"ONNX fp32 parity vs PyTorch: {parity}")
    result = {"fp32_parity": parity, "files": {"fp32": str(fp32)}}
    if ctx.cfg["export"].get("quantize_int8", True):
        try:
            int8 = quantize_int8(fp32, out / "model_int8.onnx")
            result["files"]["int8"] = str(int8)
            result["int8_parity"] = parity_report_onnx(fp32, int8, sample, tasks)
            ctx.log.info(f"int8 agreement with fp32: {result['int8_parity']}")
        except ImportError as e:
            ctx.log.warning(f"int8 quantisation skipped: {e}")
    meta = {"tasks": tasks, "spec": _spec_dict(ctx), "hop_seconds": ctx.cfg["data"]["circor_hop_seconds"],
            "per_task": {}}
    eval_summary = ctx.paths["eval"] / "summary.json"
    ev = json.loads(eval_summary.read_text())["tasks"] if eval_summary.exists() else {}
    for t in tasks:
        names = ctx.cfg["data"]["class_names"][t]
        entry = {"class_names": names,
                 "aggregation": get_aggregation(ctx.paths["eval"], t, ctx.cfg["eval"]["aggregation"]),
                 "temperature": None, "operating_threshold": None}
        if t in ev:
            cal = ev[t]["calibration"]
            entry["temperature"] = cal.get("temperature")
            op = ev[t].get("operating_sensitivity", {})
            entry["operating_threshold"] = op.get("threshold")
        meta["per_task"][t] = entry
    write_json(out / "meta.json", meta)
    write_json(out / "export_report.json", result)
    ctx.log.info(f"export written to {out}")


def _sample_batch(windows, ctx, task, n: int = 16):
    from .dataset import WindowDataset
    from .pipeline import data_root
    w = windows[f"{task}_test"].head(n)
    ds = WindowDataset(w, data_root(ctx.cfg), ctx.spec, ctx.cfg["features"], training=False)
    return np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)


def parity_report_onnx(a: Path, b: Path, sample, tasks):
    from .export import _session, run_onnx
    ra = run_onnx(_session(a), sample.numpy(), tasks)
    rb = run_onnx(_session(b), sample.numpy(), tasks)
    return {t: {"prediction_agreement": float(np.mean(ra[t].argmax(1) == rb[t].argmax(1))),
                "max_abs_logit_diff": float(np.max(np.abs(ra[t] - rb[t])))} for t in tasks}


def cmd_device_eval(ctx):
    """Recording-level metrics computed from the exported files alone: this is what the device would produce."""
    from .dataset import WindowDataset
    from .export import device_eval
    from .pipeline import data_root
    d = ctx.data()
    out = ctx.paths["export"]
    meta = json.loads((out / "meta.json").read_text())
    models = {"onnx_fp32": out / "model_fp32.onnx"}
    if (out / "model_int8.onnx").exists():
        models["onnx_int8"] = out / "model_int8.onnx"
    rows = []
    for task in ctx.cfg["data"]["tasks"]:
        w = d["windows"][f"{task}_test"]
        ds = WindowDataset(w, data_root(ctx.cfg), ctx.spec, ctx.cfg["features"], training=False)
        batch = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
        K = len(ctx.cfg["data"]["class_names"][task])
        rule = meta["per_task"][task]["aggregation"]
        for label, path in models.items():
            rows.append(device_eval(path, label, w, {task: batch}, task, rule, K, meta["tasks"]))
            ctx.log.info(f"[{task}] {label}: {rows[-1]}")
    write_json(out / "device_eval.json", rows)


def cmd_infer(ctx):
    from .export import infer_wav
    meta = json.loads(Path(ctx.args.meta).read_text())
    res = infer_wav(Path(ctx.args.wav), Path(ctx.args.model), meta, ctx.args.task, ctx.args.rule)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_report(ctx):
    from .report import build_report
    out = build_report(ctx.cfg, ctx.paths, ctx.paths["run"] / "REPORT.md")
    ctx.log.info(f"report written to {out}")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.cmd == "infer":
        ctx = type("C", (), {})()
        from .utils import get_logger
        ctx.log = get_logger("hsd.infer", None, level="WARNING")
        ctx.args = args
        return cmd_infer(ctx)
    if args.config is None:
        sys.exit("--config is required")
    ctx = Ctx(args)
    table = {"params": cmd_params, "prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate,
             "robustness": cmd_robustness, "uncertainty": cmd_uncertainty, "explain": cmd_explain,
             "export": cmd_export, "device-eval": cmd_device_eval, "report": cmd_report}
    if args.cmd == "all":
        for c in ["prepare", "train", "evaluate", "robustness", "uncertainty", "explain", "export",
                  "device-eval", "report"]:
            ctx.log.info(f"===== {c} =====")
            table[c](ctx)
    else:
        table[args.cmd](ctx)


if __name__ == "__main__":
    main()
