"""End-to-end on synthetic data in the real folder layout: prepare, train, resume, evaluate, export, device parity."""
import json
import shutil

import numpy as np
import pytest
import torch

from hsd.config import load_config


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import make_synthetic
    from pathlib import Path
    base = tmp_path_factory.mktemp("proj")
    make_synthetic.main(base / "data")
    return base


def _cfg(project, extra=()):
    base = [f"data.root={project / 'data'}", f"data.cache_dir=cache",
            f"project.runs_dir={project / 'runs'}", f"project.checkpoint_dir={project / 'ckpt'}",
            "train.num_workers=0", "model.base_channels=8", "model.hidden_dim=16",
            "train.batch_size.yaseen=8", "train.batch_size.circor=8", "train.epochs=2",
            "train.freeze_epochs=1", "train.warmup_epochs=1", "eval.bootstrap=20", "eval.mc_samples=3",
            "eval.temperature_min_windows=50", "project.device=cpu"]
    return load_config("configs/joint_main.yaml", base + list(extra))


def test_train_then_resume_continues(project):
    from hsd.cli import Ctx
    import argparse
    args = argparse.Namespace(cmd="train", config="configs/joint_main.yaml", set=[], resume=None,
                              checkpoint=None, quiet=True)
    ctx = Ctx.__new__(Ctx)
    ctx.args = args
    ctx.cfg = _cfg(project)
    from hsd.config import run_paths
    ctx.paths = run_paths(ctx.cfg)
    ctx.paths["run"].mkdir(parents=True, exist_ok=True)
    from hsd.utils import get_logger, pick_device, set_seed
    ctx.log = get_logger("hsd.test.int", None, "WARNING")
    set_seed(42)
    ctx.device = pick_device("cpu")
    ctx._data = None
    from hsd.cli import cmd_train
    cmd_train(ctx)
    h1 = json.loads((ctx.paths["run"] / "history.json").read_text())
    assert len(h1) == 2
    ctx.args = argparse.Namespace(**{**vars(args), "resume": "auto"})
    ctx.cfg = _cfg(project, ["train.epochs=3"])
    cmd_train(ctx)
    h2 = json.loads((ctx.paths["run"] / "history.json").read_text())
    assert [h["epoch"] for h in h2] == [1, 2, 3]          # continued, not restarted
    assert h2[0]["score"] == pytest.approx(h1[0]["score"])


def test_device_features_match_training_features(project):
    """The device path must produce exactly the features the model was trained on."""
    from hsd.audio import AudioSpec, bandpass, fixed_segment, log_mel, read_mono_resampled
    from hsd.dataset import WindowDataset
    from hsd.labels import build_yaseen_index, expand_windows
    from hsd.pipeline import data_root
    from hsd.config import load_config as lc
    cfg = _cfg(project)
    spec = AudioSpec(sr=4000, segment_seconds=5.0, representation="logmel", n_mels=64, n_fft=256, hop=128,
                     win=256, fmin=20.0, fmax=2000.0, bandpass=(25.0, 1000.0))
    idx = build_yaseen_index(data_root(cfg), cfg["data"]["yaseen_dir"], ["AS"], 0.5)
    w = expand_windows(idx, 5.0, None)
    w["row_id"] = range(len(w))
    ds = WindowDataset(w, data_root(cfg), spec, cfg["features"], training=False)
    train_x = ds[0][0].numpy()[0]
    # device-side path, written independently from the dataset class
    wav = data_root(cfg) / idx.path.iloc[0]
    y = read_mono_resampled(wav, 4000)
    y = bandpass(y, 4000, 25.0, 1000.0)
    start = int(round(float(w.start.iloc[0]) * 4000))
    seg = fixed_segment(y, start, spec.seg_samples)
    from hsd.audio import standardise
    dev_x = standardise(log_mel(seg, spec))
    assert np.allclose(train_x, dev_x, atol=1e-5)


def test_onnx_export_matches_pytorch(project):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from hsd.audio import AudioSpec
    from hsd.export import export_onnx, parity_report
    from hsd.models import build_model
    cfg = _cfg(project)
    net = build_model(cfg, {"yaseen": 5, "circor": 2}).eval()
    spec = AudioSpec()
    onnx_path = export_onnx(net, spec, project / "m.onnx")
    sample = torch.randn(4, 1, 64, 157)
    rep = parity_report(net, onnx_path, sample, ["yaseen", "circor"])
    for t in rep:
        assert rep[t]["max_abs_logit_diff"] < 1e-4
        assert rep[t]["prediction_agreement"] == 1.0


def test_gradcam_and_occlusion_shapes():
    from hsd.explain import gradcam, occlusion, deletion_test
    from hsd.models import build_model
    cfg = load_config("configs/base.yaml", ["model.base_channels=8"])
    net = build_model(cfg, {"yaseen": 5}).eval()
    x = torch.randn(1, 1, 64, 157)
    cam = gradcam(net, x.clone(), "yaseen", 0)
    occ = occlusion(net, x, "yaseen", 0, bf=16, bt=40)
    assert cam.shape == (64, 157) and occ.shape == (64, 157)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    d = deletion_test(net, x, "yaseen", 0, occ, [0.05])
    assert "5%" in d and set(d["5%"]) == {"attributed", "random"}
