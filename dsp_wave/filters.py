# dsp_wave/filters.py

from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi
import numpy as np
from config_wave import config


def build_sos(fs: int = None) -> np.ndarray:
    """
    Build stacked second-order sections for EMG filtering.

    Filter chain (in order):
    1. 5 Hz High-Pass (2nd order) - Remove DC offset and drift
    2. 90 Hz Low-Pass (4th order) - Anti-aliasing
    3. 20 Hz High-Pass (4th order) - Remove motion artifacts
    4. 60 Hz Notch - Powerline interference
    5. 120 Hz Notch - Second harmonic

    Returns:
        sos: Stacked SOS array, shape (n_sections, 6)
    """
    if fs is None:
        fs = config.fs

    # 1. DC removal high-pass at 5 Hz (2nd order)
    sos_dc = butter(N=2, Wn=5, btype='high', fs=fs, output='sos')

    # 2. Low-pass at 90 Hz (4th order)
    sos_lp = butter(N=4, Wn=90, btype='low', fs=fs, output='sos')

    # 3. High-pass at 20 Hz (4th order)
    sos_hp = butter(N=4, Wn=20, btype='high', fs=fs, output='sos')

    # 4. Notch at 60 Hz
    b60, a60 = iirnotch(w0=60, Q=30, fs=fs)
    sos_notch60 = np.array([[b60[0], b60[1], b60[2], 1.0, a60[1], a60[2]]])

    # 5. Notch at 120 Hz (only if below Nyquist)
    if 120 < fs / 2:
        b120, a120 = iirnotch(w0=120, Q=30, fs=fs)
        sos_notch120 = np.array([[b120[0], b120[1], b120[2], 1.0, a120[1], a120[2]]])
        return np.vstack([sos_dc, sos_lp, sos_hp, sos_notch60, sos_notch120])

    return np.vstack([sos_dc, sos_lp, sos_hp, sos_notch60])
