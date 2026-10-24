"""Export for the device: ONNX (fp32 and int8), parity checks against PyTorch, device-style evaluation,
and offline inference on a single WAV file using exactly the training preprocessing.

Parity: every number the device produces must match PyTorch within tolerance, and the int8 model must keep
recording-level accuracy close to fp32. Both are checked and written to the export folder.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .audio import AudioSpec, bandpass, fixed_segment, log_mel, read_mono_resampled, standardise
from .evaluate import aggregate, prob_cols
from .metrics import compute_metrics
from .utils import write_json


class _Heads(nn.Module):
    """Wraps HeartNet so one ONNX graph returns every task's logits."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.order = net.tasks

    def forward(self, x):
        f = self.net.backbone(x)
        return tuple(self.net.head(t)(f) for t in self.order)


def export_onnx(net: nn.Module, spec: AudioSpec, out: Path) -> Path:
    net.eval()
    wrapper = _Heads(net).eval()
    dummy = torch.randn(1, 1, spec.n_mels, spec.frames) if spec.representation == "logmel" \
        else torch.randn(1, 1, spec.seg_samples)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["window"], output_names=[f"logits_{t}" for t in net.tasks],
                  dynamic_axes={"window": {0: "batch"}, **{f"logits_{t}": {0: "batch"} for t in net.tasks}},
                  opset_version=17)
    try:
        torch.onnx.export(wrapper, (dummy,), str(out), dynamo=False, **kwargs)
    except TypeError:                      # older torch without the dynamo flag
        torch.onnx.export(wrapper, (dummy,), str(out), **kwargs)
    return out


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def _session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(sess, batch: np.ndarray, tasks: List[str]) -> Dict[str, np.ndarray]:
    outs = sess.run(None, {"window": batch.astype(np.float32)})
    return {t: o for t, o in zip(tasks, outs)}


def torch_logits(net: nn.Module, batch: torch.Tensor, tasks: List[str]) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        f = net.backbone(batch)
        return {t: net.head(t)(f).numpy() for t in tasks}


def parity_report(net, onnx_path: Path, sample: torch.Tensor, tasks: List[str]) -> dict:
    sess = _session(onnx_path)
    ref = torch_logits(net, sample, tasks)
    got = run_onnx(sess, sample.numpy(), tasks)
    rep = {}
    for t in tasks:
        diff = float(np.max(np.abs(ref[t] - got[t])))
        agree = float(np.mean(ref[t].argmax(1) == got[t].argmax(1)))
        rep[t] = {"max_abs_logit_diff": diff, "prediction_agreement": agree}
    return rep


def device_eval(model_path: Path, label: str, windows: pd.DataFrame, batches: Dict[str, np.ndarray],
                task: str, rule: str, K: int, tasks: List[str]) -> dict:
    """Recording-level metrics from the model file alone, plus per-window latency on this machine."""
    sess = _session(model_path)
    t0 = time.perf_counter()
    logits = run_onnx(sess, batches[task], tasks)[task]
    dt = time.perf_counter() - t0
    e = np.exp(logits - logits.max(1, keepdims=True))
    P = e / e.sum(1, keepdims=True)
    win = windows.copy().reset_index(drop=True)
    for k in range(K):
        win[f"p{k}"] = P[:, k]
    rec = aggregate(win, K, rule)
    m = compute_metrics(rec["label"], rec[prob_cols(K)].to_numpy().argmax(1), rec[prob_cols(K)].to_numpy(), K)
    return {"model": label, "task": task, "recording_macro_f1": m["macro_f1"], "recording_accuracy": m["accuracy"],
            "ms_per_window_this_machine": round(1000 * dt / max(len(P), 1), 3), "windows": int(len(P))}


def infer_wav(wav: Path, model_path: Path, meta: dict, task: str, rule: str | None = None) -> dict:
    """Offline inference on one recording. Same resampling, band-pass, log-mel and standardisation as training."""
    if task not in meta["tasks"]:
        raise ValueError(f"model has tasks {meta['tasks']}, not {task}")
    spec = AudioSpec(sr=meta["spec"]["sr"], segment_seconds=meta["spec"]["segment_seconds"],
                     representation=meta["spec"]["representation"], n_mels=meta["spec"]["n_mels"],
                     n_fft=meta["spec"]["n_fft"], hop=meta["spec"]["hop"], win=meta["spec"]["win"],
                     fmin=meta["spec"]["fmin"], fmax=meta["spec"]["fmax"],
                     bandpass=tuple(meta["spec"]["bandpass"]) if meta["spec"]["bandpass"] else None)
    pt = meta["per_task"][task]
    y = read_mono_resampled(wav, spec.sr)
    if spec.bandpass:
        y = bandpass(y, spec.sr, *spec.bandpass)
    dur = len(y) / spec.sr
    hop = meta["hop_seconds"] if task == "circor" else None
    if hop is None:
        starts = [max(0.0, (dur - spec.segment_seconds) / 2)]
    elif dur <= spec.segment_seconds:
        starts = [0.0]
    else:
        starts = list(np.arange(0.0, dur - spec.segment_seconds + 1e-9, hop))
        starts.append(dur - spec.segment_seconds)
    segs = []
    for s in starts:
        seg = fixed_segment(y, int(round(s * spec.sr)), spec.seg_samples)
        if spec.representation == "logmel":
            segs.append(standardise(log_mel(seg, spec))[None])
        else:
            segs.append(standardise(seg)[None])
    batch = np.stack(segs).astype(np.float32)
    logits = run_onnx(_session(model_path), batch, meta["tasks"])[task]
    e = np.exp(logits - logits.max(1, keepdims=True))
    P = e / e.sum(1, keepdims=True)
    if pt.get("temperature"):
        z = np.log(np.clip(P, 1e-8, 1)) / pt["temperature"]
        P = np.exp(z - z.max(1, keepdims=True))
        P = P / P.sum(1, keepdims=True)
    K = len(pt["class_names"])
    win = pd.DataFrame(P, columns=prob_cols(K))
    win["rec_id"] = "recording"
    win["path"] = str(wav)
    win["label"] = 0
    win["patient_id"] = "recording"
    the_rule = rule or pt["aggregation"]
    rec = aggregate(win, K, the_rule)
    probs = rec[prob_cols(K)].to_numpy()[0]
    out = {"file": str(wav), "task": task, "windows": len(segs), "rule": the_rule,
           "prediction": pt["class_names"][int(probs.argmax())],
           "probabilities": {n: round(float(p), 4) for n, p in zip(pt["class_names"], probs)}}
    if task == "circor" and pt.get("operating_threshold") is not None:
        out["present_at_operating_point"] = bool(probs[1] >= pt["operating_threshold"])
        out["operating_threshold"] = pt["operating_threshold"]
    return out
