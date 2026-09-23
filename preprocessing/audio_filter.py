import numpy as np
from scipy.signal import butter, sosfilt


def butterworth_bandpass(
    audio,
    sample_rate,
    lowcut=20,
    highcut=400,
    order=4
):
    """
    Apply a 4th-order Butterworth bandpass filter
    to a heart sound signal.
    """

    nyquist = sample_rate / 2

    if highcut >= nyquist:
        raise ValueError(
            f"highcut must be below Nyquist frequency ({nyquist} Hz)"
        )

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    filtered_audio = sosfilt(sos, audio)

    return filtered_audio.astype(np.float32)
