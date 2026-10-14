"""Segment a fixed-length window from a feature map."""
from __future__ import annotations

import numpy as np


def segment_feature(feature: np.ndarray,
                    segment_frames: int | None = None,
                    training: bool = False,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop or pad the time axis to `segment_frames`.

    feature : (F, T) or (T,)
    """
    if segment_frames is None or feature.shape[-1] == segment_frames:
        return feature

    t = feature.shape[-1]
    if t > segment_frames:
        if training:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, t - segment_frames + 1))
        else:
            start = (t - segment_frames) // 2
        return feature[..., start:start + segment_frames]

    pad = segment_frames - t
    left = pad // 2
    right = pad - left
    pad_width = [(0, 0)] * (feature.ndim - 1) + [(left, right)]
    return np.pad(feature, pad_width, mode="constant")