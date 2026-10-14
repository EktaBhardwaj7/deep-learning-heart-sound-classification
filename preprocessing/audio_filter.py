"""Butterworth band-pass filter for PCG signals."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butterworth_bandpass(y: np.ndarray,
                         sr: int,
                         lowcut: float = 25.0,
                         highcut: float = 400.0,
                         order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Defaults reflect typical PCG content (S1/S2 + murmurs live under ~400 Hz).
    Uses second-order sections (SOS) for numerical stability at high order.
    """
    nyq = 0.5 * sr
    highcut = min(highcut, nyq * 0.99)
    lowcut = max(lowcut, 1.0)
    if lowcut >= highcut:
        return y.astype(np.float32)

    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    try:
        filtered = sosfiltfilt(sos, y)
    except ValueError:
        # signal too short for filtfilt -> fall back to causal filt
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, y)
    return filtered.astype(np.float32)