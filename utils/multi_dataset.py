"""
Multi-dataset loaders for joint training.

Both Yaseen and CirCor are converted into Log-Mel (1, 64, 157) tensors
at 4 kHz. Each task gets its own DataLoader; the trainer alternates them.

Two regularisation / domain-adaptation mechanisms are built in:
  1. Per-sample instance normalization: each (64, 157) spectrogram is
     standardized independently. This removes hardware gain and baseline
     offset differences between Yaseen and CirCor.
  2. SpecAugment: random time/frequency masking during training.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly
import librosa


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
SR_TARGET = 4000
N_MELS = 64
N_FFT = 256
HOP = 128
WIN = 256
FMIN = 20.0
FMAX = 2000.0
SEG_SEC = 5.0
SEG_SAMPLES = int(SR_TARGET * SEG_SEC)
TARGET_FRAMES = 1 + SEG_SAMPLES // HOP    # 157


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _load_resample(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR_TARGET:
        g = gcd(int(sr), int(SR_TARGET))
        up = SR_TARGET // g
        down = int(sr) // g
        y = resample_poly(y, up, down).astype(np.float32)
    return y


def _log_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR_TARGET,
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    return np.log(mel + 1e-8).astype(np.float32)


def _fix_frames(x: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    t = x.shape[-1]
    if t == target:
        return x
    if t > target:
        start = (t - target) // 2
        return x[:, start:start + target]
    return np.pad(x, ((0, 0), (0, target - t)), mode="constant")


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------
class SpecAugment:
    """Random time and frequency masking on a (1, F, T) tensor."""

    def __init__(self, freq_mask=8, time_mask=16, num_freq=2, num_time=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_freq = num_freq
        self.num_time = num_time

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, F, T = x.shape

        for _ in range(self.num_freq):
            f = int(torch.randint(0, self.freq_mask + 1, (1,)).item())
            if f == 0 or F <= f:
                continue
            f0 = int(torch.randint(0, F - f, (1,)).item())
            x[:, f0:f0 + f, :] = 0.0

        for _ in range(self.num_time):
            t = int(torch.randint(0, self.time_mask + 1, (1,)).item())
            if t == 0 or T <= t:
                continue
            t0 = int(torch.randint(0, T - t, (1,)).item())
            x[:, :, t0:t0 + t] = 0.0

        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    """
    Labels CSV needs: filename, segment_start, label, use_for_training.
    Returns (x, y), x = (1, 64, 157), y = long scalar.

    Normalization is per-sample instance normalization (mean/std of this
    spectrogram only). This is picklable and eliminates the need for a
    globally fitted normalizer.
    """

    def __init__(self, root: Path, labels: pd.DataFrame,
                 training: bool = False,
                 per_file_cache: bool = True,
                 use_specaugment: bool = True):
        self.root = Path(root)
        self.labels = labels.reset_index(drop=True).copy()
        self.training = training
        self.per_file_cache = per_file_cache
        self._cache: dict[str, np.ndarray] = {}
        self.augment = SpecAugment() if (training and use_specaugment) else None

    def __len__(self):
        return len(self.labels)

    def _full_spec(self, filename: str) -> np.ndarray:
        if self.per_file_cache and filename in self._cache:
            return self._cache[filename]
        y = _load_resample(self.root / filename)
        spec = _log_mel(y)
        if self.per_file_cache:
            self._cache[filename] = spec
        return spec

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        full = self._full_spec(row["filename"])

        frame_start = int(round(float(row["segment_start"]) * SR_TARGET / HOP))
        frame_end = frame_start + TARGET_FRAMES

        if frame_end <= full.shape[1]:
            x = full[:, frame_start:frame_end]
        else:
            tail = full[:, frame_start:] if frame_start < full.shape[1] else full[:, :0]
            pad = TARGET_FRAMES - tail.shape[1]
            x = np.pad(tail, ((0, 0), (0, pad)), mode="constant")

        x = _fix_frames(x, TARGET_FRAMES).astype(np.float32)
        x = x[None, :, :]
        xt = torch.from_numpy(x)

        # Per-sample instance normalization
        mean = xt.mean()
        std = xt.std().clamp_min(1e-7)
        xt = (xt - mean) / std

        if self.augment is not None:
            xt = self.augment(xt)

        return xt, int(row["label"])


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n_total = counts.sum()
    weights = n_total / (num_classes * counts)
    return (weights / weights.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def patient_level_split(df: pd.DataFrame, seed: int = 42,
                        train: float = 0.70, val: float = 0.15):
    patients = np.asarray(df["patient_id"].astype(str).unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_tr = int(train * n)
    n_va = int(val * n)

    tr = set(patients[:n_tr].tolist())
    va = set(patients[n_tr:n_tr + n_va].tolist())
    te = set(patients[n_tr + n_va:].tolist())

    return (df[df["patient_id"].astype(str).isin(tr)].reset_index(drop=True),
            df[df["patient_id"].astype(str).isin(va)].reset_index(drop=True),
            df[df["patient_id"].astype(str).isin(te)].reset_index(drop=True))


def recording_level_split(df: pd.DataFrame, seed: int = 42,
                          train: float = 0.80, val: float = 0.10):
    from sklearn.model_selection import train_test_split
    recs = df.drop_duplicates(subset="filename").reset_index(drop=True)

    recs_tr, recs_temp = train_test_split(
        recs, test_size=1 - train, stratify=recs["label"], random_state=seed,
    )
    val_ratio = val / (1 - train)
    recs_va, recs_te = train_test_split(
        recs_temp, test_size=1 - val_ratio,
        stratify=recs_temp["label"], random_state=seed,
    )

    tr_files = set(recs_tr["filename"].tolist())
    va_files = set(recs_va["filename"].tolist())
    te_files = set(recs_te["filename"].tolist())

    return (df[df["filename"].isin(tr_files)].reset_index(drop=True),
            df[df["filename"].isin(va_files)].reset_index(drop=True),
            df[df["filename"].isin(te_files)].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_loaders(yaseen_root: Path, circor_root: Path,
                  yaseen_csv: Path, circor_csv: Path,
                  batch_yaseen: int = 32, batch_circor: int = 32,
                  num_workers: int = 2, seed: int = 42,
                  use_specaugment: bool = True):
    df_y = pd.read_csv(yaseen_csv)
    df_c = pd.read_csv(circor_csv)

    df_y = df_y[df_y["use_for_training"] == 1].reset_index(drop=True)
    y_tr, y_va, y_te = recording_level_split(df_y, seed=seed)

    c_train_all, c_val_all, c_test_all = patient_level_split(df_c, seed=seed)
    c_tr = c_train_all[c_train_all["use_for_training"] == 1].reset_index(drop=True)
    c_va = c_val_all[c_val_all["use_for_training"] == 1].reset_index(drop=True)
    c_te = c_test_all[c_test_all["use_for_training"] == 1].reset_index(drop=True)

    ds = {
        "yaseen_train": SpectrogramDataset(yaseen_root, y_tr, training=True,
                                           use_specaugment=use_specaugment),
        "yaseen_val":   SpectrogramDataset(yaseen_root, y_va, training=False),
        "yaseen_test":  SpectrogramDataset(yaseen_root, y_te, training=False),
        "circor_train": SpectrogramDataset(circor_root, c_tr, training=True,
                                           use_specaugment=use_specaugment),
        "circor_val":   SpectrogramDataset(circor_root, c_va, training=False),
        "circor_test":  SpectrogramDataset(circor_root, c_te, training=False),
    }

    for k, v in ds.items():
        if len(v) == 0:
            raise RuntimeError(f"Empty split: {k}")

    loaders = {
        k: DataLoader(
            v,
            batch_size=batch_yaseen if "yaseen" in k else batch_circor,
            shuffle=("train" in k),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=("train" in k),
        )
        for k, v in ds.items()
    }

    cw_circor = compute_class_weights(c_tr["label"].to_numpy(), num_classes=2)
    cw_yaseen = compute_class_weights(y_tr["label"].to_numpy(), num_classes=5)

    info = {
        "yaseen_counts": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
        "circor_counts": {"train": len(c_tr), "val": len(c_va), "test": len(c_te)},
        "circor_unknown_in_val": int((c_val_all["use_for_training"] == 0).sum()),
        "circor_unknown_in_test": int((c_test_all["use_for_training"] == 0).sum()),
        "input_shape": [1, N_MELS, TARGET_FRAMES],
        "yaseen_class_counts_train": y_tr["label"].value_counts().sort_index().tolist(),
        "circor_class_counts_train": c_tr["label"].value_counts().sort_index().tolist(),
        "yaseen_class_weights": cw_yaseen.tolist(),
        "circor_class_weights": cw_circor.tolist(),
    }
    return loaders, info
