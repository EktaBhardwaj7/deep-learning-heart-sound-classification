"""Log-Mel spectrogram extraction."""
from __future__ import annotations

import numpy as np
import librosa


def extract_log_mel(y: np.ndarray,
                    sr: int,
                    n_mels: int = 128,
                    n_fft: int = 1024,
                    hop_length: int = 256,
                    win_length: int = 512,
                    fmin: float = 20.0,
                    fmax: float = 4000.0,
                    eps: float = 1e-8) -> np.ndarray:
    """Return log-Mel spectrogram of shape (n_mels, n_frames)."""
    fmax = min(fmax, sr / 2.0)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax,
        power=2.0,
    )
    return np.log(mel + eps).astype(np.float32)