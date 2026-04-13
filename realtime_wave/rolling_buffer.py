import numpy as np


class RollingBuffer:
    """
    Maintains rolling buffer and emits overlapping windows.
    push(chunk) → (n_windows, window_size)
    """

    def __init__(self, window_size: int, hop: int, dtype=np.float32):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if hop <= 0:
            raise ValueError("hop must be positive")
        if hop > window_size:
            raise ValueError("hop must be <= window_size")

        self.window_size = window_size
        self.hop = hop
        self.dtype = dtype
        self.buffer = None

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=self.dtype)

        if chunk.shape[0] == 0:
            return np.empty((0, self.window_size, chunk.shape[1]), dtype=self.dtype)
        if chunk.ndim == 1:
            chunk = chunk[:, None]

        if self.buffer is None:
            self.buffer = chunk.astype(self.dtype)
        else:
            self.buffer = np.vstack((self.buffer, chunk))

        windows = []


        while self.buffer.shape[0] >= self.window_size:
            win = self.buffer[: self.window_size].copy()
            windows.append(win)
            self.buffer = self.buffer[self.hop :]

        if not windows:
            return np.empty((0, self.window_size, chunk.shape[1]), dtype=self.dtype)

        return np.stack(windows, axis=0)

