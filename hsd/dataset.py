"""PyTorch dataset over windows. Each item is one 5-second window: returns (x, label, row_id)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import AudioSpec, add_noise, fixed_segment, log_mel, standardise, waveform


class SpectrogramAugment:
    """SpecAugment: random frequency and time bands set to zero (0 = the mean after standardisation)."""

    def __init__(self, freq_mask=8, time_mask=16, num_freq=2, num_time=2, **_):
        self.fm, self.tm, self.nf, self.nt = freq_mask, time_mask, num_freq, num_time

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, F, T = x.shape
        for _ in range(self.nf):
            f = int(torch.randint(0, self.fm + 1, (1,)))
            if f and F > f:
                f0 = int(torch.randint(0, F - f, (1,)))
                x[:, f0:f0 + f, :] = 0.0
        for _ in range(self.nt):
            t = int(torch.randint(0, self.tm + 1, (1,)))
            if t and T > t:
                t0 = int(torch.randint(0, T - t, (1,)))
                x[:, :, t0:t0 + t] = 0.0
        return x


class WindowDataset(Dataset):
    def __init__(self, windows: pd.DataFrame, root: Path, spec: AudioSpec, cfg_features: dict,
                 training: bool, noise: Optional[Tuple[str, float]] = None, noise_seed: int = 0):
        self.w = windows.reset_index(drop=True)
        self.root = Path(root)
        self.spec = spec
        self.training = training
        self.noise = noise
        self.noise_seed = noise_seed
        self.jitter = float(cfg_features.get("time_jitter_seconds", 0.0)) if training else 0.0
        aug = cfg_features.get("specaugment", {})
        self.specaug = SpectrogramAugment(**aug) if (training and aug.get("enabled", True)
                                                     and spec.representation == "logmel") else None
        raw_aug = cfg_features.get("raw_augment", {})
        self.raw_aug = raw_aug if (training and raw_aug.get("enabled", True)) else None

    def __len__(self) -> int:
        return len(self.w)

    @property
    def labels(self) -> np.ndarray:
        return self.w["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, i: int):
        row = self.w.iloc[i]
        s = self.spec
        wav = waveform(self.root, row["path"], s)
        start = int(round(float(row["start"]) * s.sr))
        if self.jitter > 0:
            j = int(self.jitter * s.sr)
            start += int(np.random.randint(-j, j + 1))
        start = int(np.clip(start, 0, max(len(wav) - s.seg_samples, 0)))
        seg = fixed_segment(wav, start, s.seg_samples)
        if self.noise is not None:
            kind, snr = self.noise
            rng = np.random.default_rng(self.noise_seed * 1_000_003 + i)
            seg = add_noise(seg, snr, kind, rng, s)

        if s.representation == "raw":
            x = standardise(seg)
            if self.raw_aug is not None:
                g = float(self.raw_aug.get("gain", 0.2))
                x = x * (1.0 + np.random.uniform(-g, g))
                x = x + np.random.randn(len(x)).astype(np.float32) * float(self.raw_aug.get("noise_std", 0.02))
            x = torch.from_numpy(x.astype(np.float32))[None, :]
        else:
            x = torch.from_numpy(standardise(log_mel(seg, s)))[None]
            if self.specaug is not None:
                x = self.specaug(x)
        return x, int(row["label"]), int(row["row_id"])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    def _init(worker_id):
        s = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(s)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(ds) > batch_size,
                      persistent_workers=num_workers > 0, worker_init_fn=_init, generator=g)
