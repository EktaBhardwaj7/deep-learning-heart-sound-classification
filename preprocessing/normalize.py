"""Normalization utilities."""
from __future__ import annotations

import numpy as np


def z_score_normalize(feature: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Global z-score normalization.

    Accepts (T,) waveform or (F, T) / (N, F, T) feature. Computes a single
    mean/std over the whole array; for per-channel stats use
    FeatureNormalizer from utils.dataset.
    """
    mean = feature.mean()
    std = feature.std()
    return ((feature - mean) / (std + eps)).astype(np.float32)