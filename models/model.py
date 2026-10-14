"""
Model architectures for heart-sound classification.

All models return raw logits (B, num_classes). Softmax is applied
inside the loss function, not inside the model.

Raw waveform input:  (B, 1, T)
Log-Mel input:       (B, 1, F, T)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def _init_module(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 1D blocks
# ---------------------------------------------------------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pool=2, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.drop(F.relu(self.bn(self.conv(x)))))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        return F.relu(out + identity)


# ---------------------------------------------------------------------------
# 2D blocks
# ---------------------------------------------------------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return self.pool(x)


class SEBlock2D(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    """1D CNN for raw waveform. Input: (B, 1, T)."""
    def __init__(self, num_classes=5, in_channels=1, dropout=0.3, base_channels=32, **_):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(
            ConvBlock1D(in_channels, b, kernel=15, pool=4, dropout=dropout * 0.5),
            ConvBlock1D(b, b * 2, kernel=11, pool=4, dropout=dropout * 0.5),
            ConvBlock1D(b * 2, b * 4, kernel=7, pool=4, dropout=dropout),
            ConvBlock1D(b * 4, b * 4, kernel=5, pool=2, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(b * 4, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class ResNet1D(nn.Module):
    """Deeper 1D residual network for raw waveform. Input: (B, 1, T)."""
    def __init__(self, num_classes=5, in_channels=1, dropout=0.3, base_channels=32, **_):
        super().__init__()
        b = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, b, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(b),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(b, b, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(b, b * 2, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(b * 2, b * 4, blocks=2, stride=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(b * 4, num_classes),
        )

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride, dropout):
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class CNN2D(nn.Module):
    """Baseline 2D CNN for Log-Mel. Input: (B, 1, F, T)."""
    def __init__(self, num_classes=5, in_channels=1, dropout=0.3, base_channels=32, **_):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(
            ConvBlock2D(in_channels, b, dropout=dropout * 0.5),
            ConvBlock2D(b, b * 2, dropout=dropout * 0.5),
            ConvBlock2D(b * 2, b * 4, dropout=dropout),
            ConvBlock2D(b * 4, b * 4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(b * 4, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class AttentionCNN2D(nn.Module):
    """CNN2D with channel attention. Input: (B, 1, F, T)."""
    def __init__(self, num_classes=5, in_channels=1, dropout=0.3, base_channels=32, **_):
        super().__init__()
        b = base_channels
        self.features = nn.Sequential(
            ConvBlock2D(in_channels, b), SEBlock2D(b),
            ConvBlock2D(b, b * 2), SEBlock2D(b * 2),
            ConvBlock2D(b * 2, b * 4), SEBlock2D(b * 4),
            ConvBlock2D(b * 4, b * 4), SEBlock2D(b * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(b * 4, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CRNN(nn.Module):
    """CNN2D encoder + BiLSTM. Input: (B, 1, F, T)."""
    def __init__(self, num_classes=5, in_channels=1, dropout=0.3,
                 base_channels=32, hidden_dim=128, num_layers=2, **_):
        super().__init__()
        b = base_channels
        self.cnn = nn.Sequential(
            ConvBlock2D(in_channels, b, dropout=dropout * 0.5),
            ConvBlock2D(b, b * 2, dropout=dropout * 0.5),
            ConvBlock2D(b * 2, b * 4, dropout=dropout),
            ConvBlock2D(b * 4, b * 4, dropout=dropout),
        )
        self.rnn = nn.LSTM(
            input_size=b * 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)             # (B, C, F', T')
        x = x.mean(dim=2)           # (B, C, T')
        x = x.transpose(1, 2)       # (B, T', C)
        out, _ = self.rnn(x)
        out = out.mean(dim=1)       # (B, 2H)
        return self.head(out)


class BiLSTMOnly(nn.Module):
    """BiLSTM over Log-Mel frames. Input: (B, 1, F, T)."""
    def __init__(self, num_classes=5, n_mels=128, dropout=0.3,
                 hidden_dim=128, num_layers=2, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)   # (B, T, F)
        x = F.relu(self.input_proj(x))
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.head(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder over Log-Mel frames. Input: (B, 1, F, T)."""
    def __init__(self, num_classes=5, n_mels=128, dropout=0.2,
                 hidden_dim=128, num_layers=3, nhead=4,
                 dim_feedforward=256, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)   # (B, T, F)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "cnn2d": CNN2D,
    "attention_cnn2d": AttentionCNN2D,
    "crnn": CRNN,
    "bilstm": BiLSTMOnly,
    "transformer": TransformerClassifier,
}


def build_model(name: str, num_classes: int = 5, n_mels: int = 128, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if name in ("bilstm", "transformer"):
        kwargs.setdefault("n_mels", n_mels)
    model = cls(num_classes=num_classes, **kwargs)
    model.apply(_init_module)
    return model