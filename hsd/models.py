"""Backbones, temporal pooling and the two-head network.

Backbone outputs a feature vector (B, d). A HeartNet has one backbone and one linear head per task,
so a single-task run and the joint model use exactly the same code.

Backbones
  cnn1d, resnet1d      raw waveform input (B, 1, L)
  cnn2d, resnet2d      log-mel input (B, 1, F, T); resnet2d is a ResNet-18 style network
  crnn                 2-D convolution, then BiLSTM over time
  bilstm               BiLSTM directly over log-mel frames
  transformer          Transformer encoder over log-mel frames
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AttentionPool(nn.Module):
    """Learned weight per time step (softmax over T). Weights are kept for explanation."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        self.last_weights = w.detach()
        return (x * w.unsqueeze(-1)).sum(dim=1)


class MeanPool(nn.Module):
    last_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


def temporal_pool(kind: str, dim: int) -> nn.Module:
    if kind == "attention":
        return AttentionPool(dim)
    if kind == "mean":
        return MeanPool()
    raise ValueError(kind)


# ---------------------------------------------------------------- 1-D (raw waveform)
class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k, pool, dropout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.drop(F.relu(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    input_kind = "1d"

    def __init__(self, base_channels=32, dropout=0.3, **_):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(
            ConvBlock1D(1, b, 15, 4, dropout * 0.5), ConvBlock1D(b, 2 * b, 11, 4, dropout * 0.5),
            ConvBlock1D(2 * b, 4 * b, 7, 4, dropout), ConvBlock1D(4 * b, 4 * b, 5, 2, dropout))
        self.out_dim = 4 * b

    def forward(self, x):
        return F.adaptive_avg_pool1d(self.features(x), 1).flatten(1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.short = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                      if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


class ResNet1D(nn.Module):
    input_kind = "1d"

    def __init__(self, base_channels=24, block_dropout=0.1, **_):
        super().__init__()
        b = base_channels
        self.stem = nn.Sequential(nn.Conv1d(1, b, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(b),
                                  nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock1D(b, b, 1, block_dropout), ResBlock1D(b, b, 1, block_dropout))
        self.layer2 = nn.Sequential(ResBlock1D(b, 2 * b, 2, block_dropout), ResBlock1D(2 * b, 2 * b, 1, block_dropout))
        self.layer3 = nn.Sequential(ResBlock1D(2 * b, 4 * b, 2, block_dropout), ResBlock1D(4 * b, 4 * b, 1, block_dropout))
        self.layer4 = nn.Sequential(ResBlock1D(4 * b, 8 * b, 2, block_dropout), ResBlock1D(8 * b, 8 * b, 1, block_dropout))
        self.out_dim = 8 * b

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return F.adaptive_avg_pool1d(x, 1).flatten(1)


# ---------------------------------------------------------------- 2-D (log-mel)
class ConvBlock2D(nn.Module):
    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))))
        return F.max_pool2d(self.drop(x), 2)


class CNN2D(nn.Module):
    input_kind = "2d"

    def __init__(self, base_channels=32, dropout=0.3, **_):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(ConvBlock2D(1, b, dropout * 0.5), ConvBlock2D(b, 2 * b, dropout * 0.5),
                                      ConvBlock2D(2 * b, 4 * b, dropout), ConvBlock2D(4 * b, 4 * b, dropout))
        self.out_dim = 4 * b

    @property
    def cam_layer(self):
        return self.features[-1]

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class ResBlock2D(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
                      if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


class ResNet2D(nn.Module):
    """ResNet-18 style network with 2-stage stem. Layer names match the earlier v1 checkpoints."""
    input_kind = "2d"

    def __init__(self, base_channels=32, block_dropout=0.1, **_):
        super().__init__()
        b = base_channels
        self.stem = nn.Sequential(nn.Conv2d(1, b, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(b),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock2D(b, b, 1, block_dropout), ResBlock2D(b, b, 1, block_dropout))
        self.layer2 = nn.Sequential(ResBlock2D(b, 2 * b, 2, block_dropout), ResBlock2D(2 * b, 2 * b, 1, block_dropout))
        self.layer3 = nn.Sequential(ResBlock2D(2 * b, 4 * b, 2, block_dropout), ResBlock2D(4 * b, 4 * b, 1, block_dropout))
        self.layer4 = nn.Sequential(ResBlock2D(4 * b, 8 * b, 2, block_dropout), ResBlock2D(8 * b, 8 * b, 1, block_dropout))
        self.out_dim = 8 * b

    @property
    def cam_layer(self):
        return self.layer4

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class CRNN(nn.Module):
    input_kind = "2d"

    def __init__(self, base_channels=32, dropout=0.3, hidden_dim=128, num_layers=2, pooling="mean", **_):
        super().__init__()
        b = base_channels
        self.cnn = nn.Sequential(ConvBlock2D(1, b, dropout * 0.5), ConvBlock2D(b, 2 * b, dropout * 0.5),
                                 ConvBlock2D(2 * b, 4 * b, dropout), ConvBlock2D(4 * b, 4 * b, dropout))
        self.rnn = nn.LSTM(4 * b, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.pool = temporal_pool(pooling, 2 * hidden_dim)
        self.out_dim = 2 * hidden_dim

    @property
    def cam_layer(self):
        return self.cnn[-1]

    def forward(self, x):
        x = self.cnn(x).mean(dim=2).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.norm(self.pool(out))


class BiLSTM(nn.Module):
    input_kind = "2d"

    def __init__(self, n_mels=64, dropout=0.3, hidden_dim=128, num_layers=2, pooling="mean", **_):
        super().__init__()
        self.proj = nn.Linear(n_mels, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.pool = temporal_pool(pooling, 2 * hidden_dim)
        self.out_dim = 2 * hidden_dim

    def forward(self, x):
        x = F.relu(self.proj(x.squeeze(1).transpose(1, 2)))
        out, _ = self.rnn(x)
        return self.norm(self.pool(out))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class Transformer(nn.Module):
    input_kind = "2d"

    def __init__(self, n_mels=64, dropout=0.2, hidden_dim=128, num_layers=3, nhead=4,
                 dim_feedforward=256, pooling="mean", **_):
        super().__init__()
        self.proj = nn.Linear(n_mels, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = temporal_pool(pooling, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x):
        x = self.pos(self.proj(x.squeeze(1).transpose(1, 2)))
        return self.norm(self.pool(self.encoder(x)))


BACKBONES = {"cnn1d": CNN1D, "resnet1d": ResNet1D, "cnn2d": CNN2D, "resnet2d": ResNet2D,
             "crnn": CRNN, "bilstm": BiLSTM, "transformer": Transformer}


class HeartNet(nn.Module):
    """Shared backbone + one `Dropout -> Linear` head per task (head_<task>)."""

    def __init__(self, backbone: nn.Module, task_classes: Dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.task_classes = dict(task_classes)
        self.input_kind = backbone.input_kind
        for t, k in self.task_classes.items():
            setattr(self, f"head_{t}", nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.out_dim, k)))

    @property
    def tasks(self):
        return list(self.task_classes)

    def head(self, task: str) -> nn.Module:
        return getattr(self, f"head_{task}")

    def forward(self, x, task: Optional[str] = None):
        if task is None:
            if len(self.task_classes) != 1:
                raise ValueError("several heads: pass task=")
            task = self.tasks[0]
        return self.head(task)(self.backbone(x))

    def set_backbone_trainable(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(flag)

    @property
    def cam_layer(self):
        return getattr(self.backbone, "cam_layer", None)


def build_model(cfg: dict, task_classes: Dict[str, int]) -> HeartNet:
    m = cfg["model"]
    cls = BACKBONES[m["name"]]
    kw = dict(base_channels=m["base_channels"], dropout=m["dropout"], block_dropout=m.get("block_dropout", 0.1),
              hidden_dim=m.get("hidden_dim", 128), num_layers=m.get("num_layers", 2), nhead=m.get("nhead", 4),
              dim_feedforward=m.get("dim_feedforward", 256), pooling=m.get("pooling", "mean"),
              n_mels=cfg["features"]["n_mels"])
    net = HeartNet(cls(**kw), task_classes, dropout=m["dropout"])
    net.apply(_init)
    return net


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
