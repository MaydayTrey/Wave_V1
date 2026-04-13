import joblib
import os
import pandas as pd
from config_wave import config
import numpy as np

def load_trained_model(path="wave_model.joblib"):
    model = joblib.load(path)
    print(f"✅ Loaded model from {path}")
    return model


print(os.getcwd())
def load_emg_csv(path, column='emg'):
    df = pd.read_csv(path) #Establishes df as a table in memory. Uses CWD.
    #signal = df.iloc[:, 1:5].values.astype(np.float32)
    signal = df.values.astype(np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[1] != config.n_channels:
        raise ValueError(f"Expected {config.n_channels} channels, but got {signal.shape[1]}")
    return signal
