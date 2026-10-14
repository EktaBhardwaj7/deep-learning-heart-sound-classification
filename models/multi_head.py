"""
Shared-backbone multi-task model for Yaseen (5-class) + CirCor (murmur).

Both tasks consume Log-Mel (1, 64, T) tensors at 4 kHz.
A single 2D ResNet backbone extracts features; two task heads produce logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        idt = self.short(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt)


class ResNet18_2D(nn.Module):
    """Compact ResNet-18-style 2D backbone for spectrogram input (1, F, T)."""

    def __init__(self, in_channels: int = 1, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make(base,     base,     blocks=2, stride=1)
        self.layer2 = self._make(base,     base * 2, blocks=2, stride=2)
        self.layer3 = self._make(base * 2, base * 4, blocks=2, stride=2)
        self.layer4 = self._make(base * 4, base * 8, blocks=2, stride=2)
        self.out_dim = base * 8

    @staticmethod
    def _make(in_ch, out_ch, blocks, stride):
        layers = [ResBlock2D(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock2D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class MultiTaskModel(nn.Module):
    """Shared backbone + two heads. Call model(x, task='yaseen'|'circor')."""

    def __init__(self,
                 num_yaseen: int = 5,
                 num_circor: int = 2,
                 in_channels: int = 1,
                 base: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = ResNet18_2D(in_channels=in_channels, base=base)
        d = self.backbone.out_dim

        self.head_yaseen = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_yaseen))
        self.head_circor = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_circor))

    def forward(self, x, task: str):
        feats = self.backbone(x)
        if task == "yaseen":
            return self.head_yaseen(feats)
        if task == "circor":
            return self.head_circor(feats)
        raise ValueError(f"Unknown task: {task}")

    def both_heads(self, x):
        feats = self.backbone(x)
        return self.head_yaseen(feats), self.head_circor(feats)