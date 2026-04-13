# data_wave/dataset_builder.py

import numpy as np
from pathlib import Path
from scipy.signal import sosfilt, sosfilt_zi

from config_wave import config
from dsp_wave.filters import build_sos
from features_wave.generator import Feature_Generator_from_windows


def load_csv(filepath):
    """Load EMG data from CSV file."""
    filepath = Path(filepath)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)

    if data.shape[1] >= 5:
        emg_data = data[:, 1:5]
    elif data.shape[1] == 4:
        emg_data = data[:, :4]
    else:
        raise ValueError(f"Unexpected CSV format: {data.shape[1]} columns")

    return emg_data.astype(np.float64)


def filter_offline(data, sos):
    """
    Apply causal filtering for offline processing.

    Uses sosfilt (not sosfiltfilt) to match real-time behavior.
    Initializes filter state to steady-state for first sample to avoid startup transient.
    """
    n_channels = data.shape[1]
    n_sections = sos.shape[0]

    # Initialize filter state to steady-state based on first sample
    zi_base = sosfilt_zi(sos)  # Shape: (n_sections, 2)

    # Broadcast to all channels: (n_sections, 2, n_channels)
    zi = np.zeros((n_sections, 2, n_channels))
    for ch in range(n_channels):
        zi[:, :, ch] = zi_base * data[0, ch]

    # Apply causal filter
    filtered, _ = sosfilt(sos, data, axis=0, zi=zi)

    return filtered


def compute_global_stats(file_label_pairs):
    """
    Pass 1: Compute global mean and std across all files for normalization.
    """
    all_samples = []
    sos = build_sos()

    for filepath, label in file_label_pairs:
        raw_data = load_csv(filepath)
        filtered_data = filter_offline(raw_data, sos)
        all_samples.append(filtered_data)

    all_data = np.vstack(all_samples)

    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)
    global_std = np.where(global_std < 1e-6, 1.0, global_std)

    print(f"Global stats computed from {len(all_data)} samples")
    print(f"  Mean: {global_mean}")
    print(f"  Std:  {global_std}")

    return global_mean, global_std


def extract_windows(data, window_size, hop_size):
    """Extract sliding windows from data."""
    windows = []
    n_samples = len(data)

    for start in range(0, n_samples - window_size + 1, hop_size):
        window = data[start:start + window_size]
        windows.append(window)

    return windows


def build_dataset(file_label_pairs):
    """
    Build dataset from list of (filepath, label) pairs.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels array (n_samples,)
        groups: Group IDs for cross-validation (n_samples,)
        global_mean: Per-channel mean for normalization
        global_std: Per-channel std for normalization
    """
    window_size = config.window_size
    hop_size = config.hop

    print("\nPass 1: Computing global statistics...")
    global_mean, global_std = compute_global_stats(file_label_pairs)

    print("\nPass 2: Extracting features...")

    X_list = []
    y_list = []
    groups_list = []

    sos = build_sos()

    for file_idx, (filepath, label) in enumerate(file_label_pairs):
        raw_data = load_csv(filepath)
        filtered_data = filter_offline(raw_data, sos)
        normalized_data = (filtered_data - global_mean) / global_std

        windows = extract_windows(normalized_data, window_size, hop_size)

        if len(windows) == 0:
            print(f"Warning: No windows extracted from {filepath}")
            continue

        windows_array = np.array(windows)
        features = Feature_Generator_from_windows(windows_array)

        for i in range(len(features)):
            X_list.append(features[i])
            y_list.append(label)
            groups_list.append(file_idx)

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)

    print(f"\nDataset built: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels distribution:")
    for label in sorted(set(y)):
        count = np.sum(y == label)
        print(f"  {label}: {count}")

    # FIXED: Return normalization stats for real-time parity
    return X, y, groups, global_mean, global_std
