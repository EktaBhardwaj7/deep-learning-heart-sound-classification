import numpy as np


def z_score_normalize(feature):
    """
    Apply Z-score normalization to a feature matrix.
    """

    mean = np.mean(feature)
    std = np.std(feature)

    normalized = (feature - mean) / (std + 1e-8)

    return normalized.astype(np.float32)
