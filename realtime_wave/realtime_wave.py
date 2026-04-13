# realtime_wave/realtime_wave.py
"""
Real-time EMG processing pipeline.
Filter -> REST Gate -> Normalize -> Buffer -> Features -> Inference -> Cogency -> Persistence
"""

import numpy as np
from realtime_wave.rolling_buffer import RollingBuffer
from realtime_wave.streaming_filter import StreamingSOSFilter
from dsp_wave.filters import build_sos
from config_wave.invariants import config
from features_wave.generator import Feature_Generator_from_windows
from realtime_wave.cogency import cogency_fusion
from realtime_wave.persistence import temporal_persistence

# Sentinel value for REST gate output - distinct from UNCERTAIN (-1)
REST_GATE_IDX = -2


class RealTimeWave:
    def __init__(self, knn_model, logreg_model, global_mean, global_std,
                 cogency_threshold=0.70, rest_threshold=None):

        sos = build_sos(config.fs)
        self.streaming_filter = StreamingSOSFilter(sos)
        self.buffer = RollingBuffer(config.window_size, config.hop)

        self.global_mean = np.asarray(global_mean).reshape(1, -1)
        self.global_std = np.asarray(global_std).reshape(1, -1)

        self.knn_model = knn_model
        self.logreg_model = logreg_model
        self.cogency_threshold = cogency_threshold

        # REST gate threshold in normalized RMS space
        # If None, gate is disabled (fallback safe default)
        self.rest_threshold = rest_threshold

        self.persistence = temporal_persistence(window_length=3)
        self.sample_count = 0
        self.warmup_samples = config.window_size * 4  # 400 samples = 2s

        print(f"  Cogency threshold: {cogency_threshold}")
        print(f"  REST gate threshold: {rest_threshold:.4f}" if rest_threshold is not None
              else "  REST gate: DISABLED (no threshold set)")
        print(f"  Warmup: {self.warmup_samples} samples ({self.warmup_samples / config.fs:.1f}s)")

    def normalize(self, chunk: np.ndarray) -> np.ndarray:
        return (chunk - self.global_mean) / (self.global_std + 1e-8)

    def _is_rest(self, normalized_chunk: np.ndarray) -> bool:
        """
        Returns True if the chunk energy is below the REST gate threshold.
        Operates on already-normalized data.
        RMS computed per channel, then averaged across channels.
        """
        if self.rest_threshold is None:
            return False
        rms_per_channel = np.sqrt(np.mean(normalized_chunk ** 2, axis=0))
        mean_rms = float(np.mean(rms_per_channel))
        return mean_rms < self.rest_threshold

    def process_chunk(self, raw_chunk: np.ndarray):
        raw_chunk = np.atleast_2d(raw_chunk)
        if raw_chunk.shape[1] != config.n_channels:
            if raw_chunk.shape[0] == config.n_channels:
                raw_chunk = raw_chunk.T

        self.sample_count += raw_chunk.shape[0]

        # 1. Filter
        filtered = self.streaming_filter.process(raw_chunk)

        # 2. Normalize
        normalized = self.normalize(filtered)

        # 3. REST GATE - check energy before model sees anything
        if self._is_rest(normalized):
            # Signal is quiet - bypass model entirely
            # Return one REST result per hop to maintain output cadence
            n_outputs = max(1, raw_chunk.shape[0] // config.hop)
            return [(REST_GATE_IDX, 1.0)] * n_outputs

        # 4. Buffer into windows
        windows = self.buffer.push(normalized)

        if windows.shape[0] == 0:
            return []

        n_windows = windows.shape[0]

        # 5. Warmup check
        if self.sample_count < self.warmup_samples:
            return [(-1, 0.0)] * n_windows

        # 6. Extract features
        features = Feature_Generator_from_windows(windows)

        # 7. Model inference
        knn_proba = self.knn_model.predict_proba(features)
        logreg_proba = self.logreg_model.predict_proba(features)

        # 8. Cogency fusion
        labels, confidences = cogency_fusion(knn_proba, logreg_proba)

        # 9. Threshold filtering
        labels = labels.copy()
        labels[confidences < self.cogency_threshold] = -1

        # 10. Temporal persistence
        stable_labels = self.persistence.update(labels)

        return [(stable_labels[i], confidences[i]) for i in range(len(stable_labels))]

    def reset(self):
        self.streaming_filter.reset()
        self.buffer = RollingBuffer(config.window_size, config.hop)
        self.persistence = temporal_persistence(window_length=3)
        self.sample_count = 0