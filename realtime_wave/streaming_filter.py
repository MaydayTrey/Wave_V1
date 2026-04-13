from scipy.signal import butter, sosfilt, sosfilt_zi
import numpy as np


class StreamingSOSFilter:
    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=float)

        if self.sos.ndim != 2 or self.sos.shape[1] != 6:
            raise ValueError("SOS must have shape (n_sections, 6)")

        self._zi_base = sosfilt_zi(self.sos)
        self.zi = None
        self._initialized = False

    def process(self, x_chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(x_chunk, dtype=float)

        if x.size == 0:
            return x
        if x.ndim == 1:
            x = x[:, None]
        n_samples, n_channels = x.shape

        if not self._initialized:
            # Initialize to ZERO - let the high-pass naturally reject DC
            # Don't track the initial DC offset
            self.zi = np.zeros((self._zi_base.shape[0], n_channels, 2))
            self._initialized = True

        y = np.zeros_like(x)
        for ch in range(n_channels):
            y[:, ch], self.zi[:, ch] = sosfilt(
                self.sos,
                x[:, ch],
                zi=self.zi[:, ch]
            )
        return y

    def reset(self):
        """Reset filter state for new session."""
        self.zi = None
        self._initialized = False
