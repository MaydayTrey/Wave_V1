# temporal.py - UPDATED
import numpy as np
from scipy.signal import hilbert


def temporal_features(windowed_signal):
    # Existing features (keep these)
    rms = np.sqrt(np.mean(windowed_signal ** 2))
    mav = np.mean(np.abs(windowed_signal))
    wl = np.sum(np.abs(np.diff(windowed_signal)))
    zcr = np.sum(np.diff(np.sign(windowed_signal)) != 0)

    # FIXED: Compute envelope, then slope
    envelope = np.abs(hilbert(windowed_signal))

    # Envelope slope: captures if activation is INCREASING or DECREASING
    env_slope = np.polyfit(np.arange(len(envelope)), envelope, 1)[0]

    # Envelope trajectory: split window into thirds
    third = len(envelope) // 3
    env_start = np.mean(envelope[:third])
    env_mid = np.mean(envelope[third:2 * third])
    env_end = np.mean(envelope[2 * third:])

    # Trajectory ratios (direction indicators)
    eps = 1e-10
    env_trend = (env_end - env_start) / (mav + eps)  # Normalized rise/fall

    temporal_stack = np.array([
        rms, mav, wl, zcr,
        env_slope,  # NEW: envelope direction
        env_trend,  # NEW: normalized start-to-end change
    ])
    return temporal_stack
