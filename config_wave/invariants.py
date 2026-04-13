from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class WaveInvariants:
    STRICT_MODE: bool = False
    fs: int = 200
    window_size: int = 100
    hop: int = 50
    seconds: int = 2
    chunk_size: int = hop
    n_channels: int = 4
    SIM_SEED: int = 42

    samples: int = field(init=False)
    time: np.ndarray = field(init=False)
    nyquist: int = field(init=False)

    bands: tuple = (
        (20, 50),
        (50, 90),
    )

    CHANNEL_MAP = {
        0: "radial",
        1: "dorsal",
        2: "ulnar",
        3: "ventral"
    }

    def __post_init__(self):
        object.__setattr__(self, "nyquist", self.fs // 2)
        object.__setattr__(self, "samples", self.seconds * self.fs)
        object.__setattr__(self, "time", np.arange(self.samples) / self.fs)

config = WaveInvariants()