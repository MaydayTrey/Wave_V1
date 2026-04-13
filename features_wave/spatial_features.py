# spatial_features.py - UPDATED WITH NORMALIZED RATIOS
import numpy as np
from scipy.signal import hilbert


def spatial_features(window):
    # Compute envelope per channel, THEN take means
    envelopes = np.abs(hilbert(window, axis=0))
    means = np.mean(envelopes, axis=0)  # Shape: (4,)

    spatial = []
    eps = 1e-10

    # Adjacent differences (Direction)
    spatial.append(means[0] - means[1])  # D12
    spatial.append(means[1] - means[2])  # D23
    spatial.append(means[2] - means[3])  # D34

    # Opposing differences
    spatial.append(means[0] - means[2])  # D13
    spatial.append(means[1] - means[3])  # D24

    # Pairwise ratios (existing)
    spatial.append(means[0] / (means[1] + eps))  # R12
    spatial.append(means[2] / (means[3] + eps))  # R34

    # NEW: Normalized channel ratios (4 features)
    # Each channel's proportion of total envelope energy
    total_env = np.sum(means) + eps
    spatial.append(means[0] / total_env)  # NORM_CH1
    spatial.append(means[1] / total_env)  # NORM_CH2
    spatial.append(means[2] / total_env)  # NORM_CH3
    spatial.append(means[3] / total_env)  # NORM_CH4

    return np.array(spatial)  # Now 11 features
