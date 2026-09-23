import numpy as np


def segment_feature(feature, sample_rate=2000, duration=5, hop_ms=15):
    """
    Extract a fixed-duration segment from a Log-Mel feature.

    Parameters:
        feature: Log-Mel spectrogram of shape (mel_bins, time_frames)
        sample_rate: original audio sampling rate
        duration: segment duration in seconds
        hop_ms: hop length used during Log-Mel extraction

    Returns:
        Fixed-size feature segment
    """

    # Number of time frames needed for the required duration
    frames = int(duration * 1000 / hop_ms)

    # If feature is already shorter than required,
    # repeat it until it becomes long enough
    if feature.shape[1] < frames:
        repeat_count = int(np.ceil(frames / feature.shape[1]))
        feature = np.tile(feature, (1, repeat_count))

    # Take the first fixed-length segment
    segment = feature[:, :frames]

    return segment.astype(np.float32)
