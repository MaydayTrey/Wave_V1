import numpy as np
from config_wave import config

def windowing(filtered_signal, window_size, hop):
    windows = []
    n_samples = filtered_signal.shape[0]
    if filtered_signal.ndim == 1:
        filtered_signal = filtered_signal[:, None]
    for i in range(0, n_samples - window_size + 1, hop):
        windows.append(filtered_signal[i:i + window_size])
    if not windows:
        return np.empty((0,window_size, filtered_signal.shape[1]))
    return np.stack(windows, axis=0)
#Changed return np.array(windows) to np.stack() for multi-channel

# if I was to do this, I would just put the windowing in a channel loop, so for ch in channels
# do the same loop?