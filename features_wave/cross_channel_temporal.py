# cross_channel_temporal.py - CLEANED UP (removed low-discriminative timing features)
import numpy as np
from scipy.signal import hilbert


def cross_channel_temporal_features(window):
    """
    Cross-channel slope and correlation features.
    Returns 8 features.
    """
    features = []
    n_channels = window.shape[1]

    # Slopes between adjacent channels
    for i in range(n_channels - 1):
        slope = np.mean(window[:, i + 1] - window[:, i])
        features.append(slope)  # 3 features: slope_12, slope_23, slope_34

    # Correlations between adjacent channels
    for i in range(n_channels - 1):
        corr = np.corrcoef(window[:, i], window[:, i + 1])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        features.append(corr)  # 3 features: corr_12, corr_23, corr_34

    # Non-adjacent correlations
    corr_13 = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
    corr_24 = np.corrcoef(window[:, 1], window[:, 3])[0, 1]
    features.append(0.0 if np.isnan(corr_13) else corr_13)
    features.append(0.0 if np.isnan(corr_24) else corr_24)

    return features  # 8 features total


def compute_covar_feats(window):
    """
    Upper triangle of covariance matrix.
    Returns 6 features.
    """
    cov = np.cov(window.T)
    # Extract upper triangle (excluding diagonal)
    indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return [cov[i, j] for i, j in indices]

# REMOVED: cross_channel_envelope_features() - timing features showed d < 0.25
