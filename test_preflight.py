# test_preflight.py
from config_wave import config
from dsp_wave import build_sos
import joblib
import numpy as np

print(f"Sample rate: {config.fs}")
print(f"Nyquist: {config.fs / 2}")
print(f"Window: {config.window_size}, Hop: {config.hop}")
print(f"Channels: {config.n_channels}")

sos = build_sos()
print(f"SOS shape: {sos.shape}")

knn = joblib.load("models/lda_to_knn.joblib")
logreg = joblib.load("models/lda_to_logreg.joblib")
print(f"KNN classes: {knn.classes_}")
print(f"LogReg classes: {logreg.classes_}")

# Test feature count
from features_wave import Feature_Generator_from_windows
dummy = np.random.randn(1, config.window_size, config.n_channels)
feats = Feature_Generator_from_windows(dummy)
print(f"Feature count: {feats.shape[1]}")