import numpy as np
import torch

from hsd.audio import AudioSpec, add_noise, fixed_segment, make_noise
from hsd.config import load_config
from hsd.models import build_model, count_params


def test_fixed_segment_wraps_short_clips():
    y = np.arange(10, dtype=np.float32)
    s = fixed_segment(y, 0, 25)
    assert len(s) == 25 and s[10] == 0.0          # repeats the clip instead of padding zeros


def test_snr_is_exact():
    spec = AudioSpec(bandpass=None)
    rng = np.random.default_rng(0)
    seg = rng.standard_normal(20000).astype(np.float32)
    for snr in (20, 0, -5):
        noisy = add_noise(seg, snr, "white", np.random.default_rng(1), spec)
        noise = noisy - seg
        measured = 10 * np.log10(np.mean(seg ** 2) / np.mean(noise ** 2))
        assert abs(measured - snr) < 0.2


def test_noise_kinds_have_unit_scale():
    rng = np.random.default_rng(0)
    for kind in ("white", "pink", "hum"):
        x = make_noise(kind, 8000, 4000, rng)
        assert abs(float(np.std(x)) - 1.0) < 1e-3


def test_matched_comparison_sizes_from_the_proposal():
    """RQ1: the log-mel and raw backbones must be matched in size."""
    cm = build_model(load_config("configs/rq1_logmel.yaml"), {"yaseen": 5})
    rm = build_model(load_config("configs/rq1_raw.yaml"), {"yaseen": 5})
    assert count_params(cm.backbone) == 582304
    assert count_params(rm.backbone) == 543096


def test_every_backbone_has_both_heads():
    for name, rep in [("cnn1d", "raw"), ("resnet1d", "raw"), ("cnn2d", "logmel"), ("resnet2d", "logmel"),
                      ("crnn", "logmel"), ("bilstm", "logmel"), ("transformer", "logmel")]:
        cfg = load_config("configs/base.yaml", [f"model.name={name}", f"features.representation={rep}",
                                                "model.base_channels=8", "model.hidden_dim=16"])
        net = build_model(cfg, {"yaseen": 5, "circor": 2}).eval()
        x = torch.randn(3, 1, 20000) if rep == "raw" else torch.randn(3, 1, 64, 157)
        with torch.no_grad():
            assert net(x, "yaseen").shape == (3, 5)
            assert net(x, "circor").shape == (3, 2)


def test_attention_weights_sum_to_one():
    cfg = load_config("configs/base.yaml", ["model.name=crnn", "model.pooling=attention",
                                            "model.base_channels=8", "model.hidden_dim=16"])
    net = build_model(cfg, {"circor": 2}).eval()
    with torch.no_grad():
        net(torch.randn(2, 1, 64, 157), "circor")
    assert torch.allclose(net.backbone.pool.last_weights.sum(1), torch.ones(2), atol=1e-5)


def test_mean_pooling_keeps_v1_state_keys():
    cfg = load_config("configs/base.yaml", ["model.name=crnn", "model.base_channels=8", "model.hidden_dim=16"])
    net = build_model(cfg, {"circor": 2})
    assert not any(k.startswith("backbone.pool.score") for k in net.state_dict())


def test_backbone_and_head_learning_rates():
    from hsd.trainer import build_optimizer
    cfg = load_config("configs/base.yaml", ["model.base_channels=8", "train.backbone_lr_mult=0.1",
                                            "train.optimizer.lr=0.001"])
    net = build_model(cfg, {"yaseen": 5, "circor": 2})
    opt = build_optimizer(net, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert abs(lrs[0] - 1e-4) < 1e-12 and abs(lrs[1] - 1e-3) < 1e-12
