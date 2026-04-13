from config_wave import config
import numpy as np

from utils_wave.diagnostic import Diagnostic_Decorator

rng = np.random.default_rng(config.SIM_SEED)

Powerline_Interference = 0.3 * np.sin(2 * np.pi * 60 * config.time)
Harmonic_Interference = 0.02 * np.cos(2 * np.pi * 120 * config.time)

@Diagnostic_Decorator
def Signal_Generator():
    raw_noise = rng.standard_normal(config.samples)
    return raw_noise + Powerline_Interference + Harmonic_Interference

