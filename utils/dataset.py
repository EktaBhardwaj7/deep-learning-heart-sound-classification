"""
Dataset utilities:
  * load labels.csv
  * stratified train/val/test split (recording-level; Yaseen has no subject IDs)
  * extract raw waveform OR Log-Mel features, cached to disk
  * normalise using train-only statistics
  * PyTorch Dataset + DataLoader
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import librosa

from features.logmel import extract_log_mel
from preprocessing.audio_filter import butterworth_bandpass


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def build_splits(labels_df: pd.DataFrame,
                 train_split: float = 0.70,
                 val_split: float = 0.15,
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified recording-level split.

    NOTE: Yaseen does not expose subject IDs; patient-level split is not
    possible and this is documented as a limitation in the proposal.
    """
    assert 0 < train_split < 1 and 0 < val_split < 1
    assert train_split + val_split < 1

    train_df, temp_df = train_test_split(
        labels_df,
        test_size=(1.0 - train_split),
        stratify=labels_df["label"],
        random_state=seed,
    )
    val_ratio_of_temp = val_split / (1.0 - train_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_of_temp),
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------
def _load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _fix_length(y: np.ndarray, n_samples: int, training: bool) -> np.ndarray:
    if len(y) >= n_samples:
        if training:
            start = np.random.randint(0, len(y) - n_samples + 1)
        else:
            start = (len(y) - n_samples) // 2
        return y[start: start + n_samples]
    pad = n_samples - len(y)
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right), mode="constant")


def _extract_raw(path: Path, sr: int, duration: float,
                 apply_bandpass: bool, training: bool) -> np.ndarray:
    y = _load_audio(path, sr)
    if apply_bandpass:
        y = butterworth_bandpass(y, sr)
    n = int(round(duration * sr))
    y = _fix_length(y, n, training=training)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y.astype(np.float32)


def _extract_logmel(path: Path, sr: int, apply_bandpass: bool,
                    n_mels: int, n_fft: int, hop: int, win: int,
                    fmin: float, fmax: float) -> np.ndarray:
    y = _load_audio(path, sr)
    if apply_bandpass:
        y = butterworth_bandpass(y, sr)
    mel = extract_log_mel(y, sr, n_mels=n_mels, n_fft=n_fft,
                          hop_length=hop, win_length=win,
                          fmin=fmin, fmax=fmax)
    return mel.astype(np.float32)


def compute_target_frames(duration: float, sr: int, hop_length: int) -> int:
    """Number of STFT frames librosa produces for a fixed-length clip."""
    n_samples = int(round(duration * sr))
    return 1 + n_samples // hop_length


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def build_feature_cache(labels_df: pd.DataFrame,
                        dataset_root: Path,
                        representation: str,
                        sr: int, duration: float,
                        apply_bandpass: bool,
                        logmel_cfg: dict,
                        cache_path: Path,
                        logger=None):
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists():
        if logger:
            logger.info(f"Loading feature cache: {cache_path}")
        return np.load(cache_path, allow_pickle=False)["arr"], json.loads(meta_path.read_text())

    if logger:
        logger.info(f"Extracting features -> {cache_path}  (representation={representation})")

    feats: List[np.ndarray] = []
    for i, row in labels_df.iterrows():
        full = dataset_root / row["filename"]
        if representation == "raw":
            arr = _extract_raw(full, sr, duration, apply_bandpass, training=False)
        elif representation == "logmel":
            arr = _extract_logmel(full, sr, apply_bandpass, **logmel_cfg)
        else:
            raise ValueError(f"Unknown representation: {representation}")
        feats.append(arr)
        if logger and (i + 1) % 100 == 0:
            logger.info(f"  extracted {i + 1}/{len(labels_df)}")

    arr = np.stack(feats, axis=0)
    meta = {"representation": representation, "shape": list(arr.shape)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, arr=arr)
    meta_path.write_text(json.dumps(meta, indent=2))
    return arr, meta


# ---------------------------------------------------------------------------
# Normalizer (fit on train only)
# ---------------------------------------------------------------------------
class FeatureNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureNormalizer":
        if X.ndim == 2:                 # (N, T)
            mean = float(X.mean())
            std = float(X.std()) + 1e-8
        elif X.ndim == 3:               # (N, F, T) -> per-frequency bin stats
            mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            std = (X.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
        else:
            raise ValueError(f"Unsupported shape {X.shape}")
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, list]:
        return {
            "mean": np.asarray(self.mean).tolist(),
            "std": np.asarray(self.std).tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, list]) -> "FeatureNormalizer":
        return cls(mean=np.asarray(state["mean"]), std=np.asarray(state["std"]))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PCGDataset(Dataset):
    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 indices: np.ndarray,
                 representation: str,
                 target_frames: int | None,
                 augment: bool = False,
                 noise_std: float = 0.02):
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.representation = representation
        self.target_frames = target_frames
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.indices)

    def _crop_or_pad(self, x: np.ndarray) -> np.ndarray:
        if self.target_frames is None:
            return x
        t = x.shape[-1]
        if t == self.target_frames:
            return x
        if t > self.target_frames:
            if self.augment:
                start = np.random.randint(0, t - self.target_frames + 1)
            else:
                start = (t - self.target_frames) // 2
            return x[..., start:start + self.target_frames]
        pad = self.target_frames - t
        left = pad // 2
        right = pad - left
        pad_width = [(0, 0)] * (x.ndim - 1) + [(left, right)]
        return np.pad(x, pad_width, mode="constant")

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.features[idx]
        y = int(self.labels[idx])

        x = self._crop_or_pad(x)

        if self.augment:
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            gain = 1.0 + (np.random.rand() - 0.5) * 0.1
            x = x * gain

        x = np.ascontiguousarray(x.astype(np.float32))
        x = x[None, :] if self.representation == "raw" else x[None, :, :]
        return torch.from_numpy(x), y


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def make_dataloaders(labels_df: pd.DataFrame,
                     dataset_root: Path,
                     representation: str,
                     sr: int,
                     duration: float,
                     n_mels: int,
                     batch_size: int,
                     train_split: float,
                     val_split: float,
                     seed: int,
                     num_workers: int,
                     pin_memory: bool,
                     normalize: bool,
                     apply_bandpass: bool,
                     cache_dir: Path,
                     logmel_cfg: dict | None = None,
                     logger=None):
    if logmel_cfg is None:
        logmel_cfg = dict(n_mels=n_mels, n_fft=1024, hop=256, win=512, fmin=20, fmax=4000)

    cache_path = Path(cache_dir) / f"features_{representation}_sr{sr}_dur{duration}.npz"

    features, meta = build_feature_cache(
        labels_df=labels_df,
        dataset_root=Path(dataset_root),
        representation=representation,
        sr=sr, duration=duration,
        apply_bandpass=apply_bandpass,
        logmel_cfg=logmel_cfg,
        cache_path=cache_path,
        logger=logger,
    )

    labels = labels_df["label"].to_numpy(dtype=np.int64)

    train_df, val_df, test_df = build_splits(labels_df, train_split, val_split, seed)
    train_idx = train_df.index.to_numpy()
    val_idx = val_df.index.to_numpy()
    test_idx = test_df.index.to_numpy()

    if normalize:
        normalizer = FeatureNormalizer.fit(features[train_idx])
        features = normalizer.transform(features)
    else:
        normalizer = FeatureNormalizer(mean=0.0, std=1.0)

    if representation == "logmel":
        target_frames = compute_target_frames(duration, sr, logmel_cfg["hop"])
    else:
        target_frames = None

    def _ds(idx, augment):
        return PCGDataset(features, labels, idx,
                          representation=representation,
                          target_frames=target_frames,
                          augment=augment)

    train_loader = DataLoader(_ds(train_idx, True),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=False)
    val_loader = DataLoader(_ds(val_idx, False),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(_ds(test_idx, False),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    info = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "input_shape": list(features.shape[1:]),
        "representation": representation,
        "n_mels": n_mels,
        "target_frames": target_frames,
    }
    return train_loader, val_loader, test_loader, normalizer, info