import numpy as np
import librosa


def extract_log_mel(
    audio,
    sample_rate,
    mel_bins=128,
    hop_ms=15,
    win_ms=25
):
    """
    Convert heart sound waveform into Log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=mel_bins,
        hop_length=int(sample_rate * hop_ms / 1000),
        win_length=int(sample_rate * win_ms / 1000)
    )

    log_mel = np.log(mel + 1e-8)

    return log_mel.astype(np.float32)
