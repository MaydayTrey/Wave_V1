# generator.py - CLEANED UP
import numpy as np

from features_wave.spatial_features import spatial_features
from features_wave.temporal import temporal_features
from features_wave.frequency import fft_features, hanning
from features_wave.cross_channel_temporal import (
    cross_channel_temporal_features,
    compute_covar_feats
)
from config_wave import config


def Feature_Generator_from_windows(windows: np.ndarray) -> np.ndarray:
    features = []
    for window in windows:
        window_features = []

        # Per-channel features: 4 channels × 8 = 32
        for ch in range(window.shape[1]):
            signal = window[:, ch]
            temp = temporal_features(signal)  # 6 features
            hann = hanning(signal)
            freq = fft_features(hann, config.fs)  # 2 features
            window_features.extend(temp)
            window_features.extend(freq)

        # Spatial: 11 features
        spatial = spatial_features(window)
        window_features.extend(spatial)

        # Cross-channel: 8 features
        cross_temp = cross_channel_temporal_features(window)
        window_features.extend(cross_temp)

        # Covariance: 6 features
        cov_features = compute_covar_feats(window)
        window_features.extend(cov_features)

        features.append(window_features)

    return np.array(features)
