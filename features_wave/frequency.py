import numpy as np
from config_wave import config

def hanning(windowed_signal):
    #hann_weights = np.hanning(windowed_signal.shape[1])
    hann_weights = np.hanning(len(windowed_signal))
    return windowed_signal * hann_weights


def fft_features(hanned_window, fs):
    #rfft = np.fft.rfft(hanned_window, axis=1)
    rfft = np.fft.rfft(hanned_window)
    #freqs = np.fft.rfftfreq(hanned_window.shape[1], d=1 / fs)
    freqs = np.fft.rfftfreq(len(hanned_window), d=1 / fs)
    power = np.abs(rfft) ** 2

    bands_out = []

    for low, high in config.bands:
        idx = (freqs >= low) & (freqs <= high)

        if np.any(idx):
            band_power = np.mean(power[idx])
        else:
            band_power = 0.0

        bands_out.append(band_power)

    return np.array(bands_out)