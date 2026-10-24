"""Signal processing shared by training, evaluation and the device. One code path, so the device
cannot silently disagree with the desktop model."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioSpec:
    sr: int = 4000
    segment_seconds: float = 5.0
    representation: str = "logmel"
    n_mels: int = 64
    n_fft: int = 256
    hop: int = 128
    win: int = 256
    fmin: float = 20.0
    fmax: float = 2000.0
    bandpass: Optional[Tuple[float, float]] = (25.0, 1000.0)
    cache_dir: Optional[Path] = None

    @property
    def seg_samples(self) -> int:
        return int(round(self.sr * self.segment_seconds))

    @property
    def frames(self) -> int:
        return 1 + self.seg_samples // self.hop

    @property
    def cache_tag(self) -> str:
        bp = f"bp{self.bandpass[0]:g}-{self.bandpass[1]:g}" if self.bandpass else "nobp"
        return f"sr{self.sr}_{bp}"


def spec_from_cfg(cfg: dict) -> AudioSpec:
    d, f = cfg["data"], cfg["features"]
    bp = d.get("bandpass")
    return AudioSpec(
        sr=int(d["sample_rate"]), segment_seconds=float(d["segment_seconds"]),
        representation=f["representation"], n_mels=int(f["n_mels"]), n_fft=int(f["n_fft"]),
        hop=int(f["hop_length"]), win=int(f["win_length"]), fmin=float(f["fmin"]), fmax=float(f["fmax"]),
        bandpass=tuple(bp) if bp else None,
        cache_dir=Path(d.get("root", "data")) / d.get("cache_dir", "cache"),
    )


def read_mono_resampled(path: Path, sr_target: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if int(sr) != sr_target:
        g = gcd(int(sr), sr_target)
        y = resample_poly(y, sr_target // g, int(sr) // g).astype(np.float32)
    return y


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (keeps event timing for phase analysis)."""
    from scipy.signal import butter, sosfilt, sosfiltfilt
    nyq = 0.5 * sr
    hi = min(hi, 0.99 * nyq)
    if lo <= 0 or lo >= hi:
        return y.astype(np.float32)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, y).astype(np.float32)
    except ValueError:                       # too short for filtfilt padding
        return sosfilt(sos, y).astype(np.float32)


def waveform(root: Path, rel_path: str, spec: AudioSpec, use_cache: bool = True) -> np.ndarray:
    """Resampled, band-passed waveform. Cached as .npy and memory-mapped; writes are atomic."""
    cache_path = None
    if use_cache and spec.cache_dir is not None:
        safe = rel_path.replace("/", "__")
        cache_path = Path(spec.cache_dir) / spec.cache_tag / (safe + ".npy")
        if cache_path.exists():
            try:
                return np.load(cache_path, mmap_mode="r")
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)
    y = read_mono_resampled(Path(root) / rel_path, spec.sr)
    if spec.bandpass:
        y = bandpass(y, spec.sr, *spec.bandpass)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, y)
        os.replace(tmp, cache_path)
    return y


def fixed_segment(y: np.ndarray, start_sample: int, n: int) -> np.ndarray:
    """Cut n samples starting at start_sample; wrap-pad short clips (heart cycles repeat)."""
    seg = np.array(y[start_sample:start_sample + n], dtype=np.float32)
    if len(seg) == 0:
        seg = np.array(y[:n], dtype=np.float32)
    if len(seg) < n:
        reps = int(np.ceil(n / max(len(seg), 1)))
        seg = np.tile(seg, reps)[:n]
    return seg


def standardise(x: np.ndarray) -> np.ndarray:
    """Per-window standardisation, used by training AND the device. One definition, so they cannot drift apart."""
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / max(float(x.std()), 1e-7)).astype(np.float32)


def log_mel(seg: np.ndarray, spec: AudioSpec) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=seg, sr=spec.sr, n_fft=spec.n_fft, hop_length=spec.hop, win_length=spec.win,
        n_mels=spec.n_mels, fmin=spec.fmin, fmax=min(spec.fmax, spec.sr / 2), power=2.0)
    return np.log(mel + 1e-8).astype(np.float32)


NOISE_KINDS = ("white", "pink", "hum")


def make_noise(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        k = np.arange(len(spec), dtype=np.float64)
        k[0] = 1.0
        x = np.fft.irfft(spec / np.sqrt(k), n)
    elif kind == "hum":                       # 50 Hz mains (India) with harmonics
        t = np.arange(n) / sr
        x = sum(np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 2 * np.pi)) / h for h in (1, 2, 3))
    else:
        raise ValueError(f"unknown noise kind {kind}")
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def add_noise(seg: np.ndarray, snr_db: float, kind: str, rng: np.random.Generator, spec: AudioSpec) -> np.ndarray:
    """Add noise at an exact SNR. Noise is band-passed like the signal, so the SNR is measured on what the model sees."""
    noise = make_noise(kind, len(seg), spec.sr, rng)
    if spec.bandpass:
        noise = bandpass(noise, spec.sr, *spec.bandpass)
    ps = float(np.mean(seg.astype(np.float64) ** 2)) + 1e-12
    pn = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    return (seg + scale * noise).astype(np.float32)
