# diagnostic_envelope.py
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from pathlib import Path


def diagnose_envelope_timing(csv_path: str):
    """Visualize envelope timing for a single gesture file."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # Take middle window (skip transients)
    mid = len(data) // 2
    window = data[mid - 50:mid + 50, :]  # 100 samples centered

    n_samples, n_channels = window.shape
    envelopes = np.abs(hilbert(window, axis=0))

    # Compute timing
    peak_times = np.argmax(envelopes, axis=0)
    onset_times = []
    for ch in range(n_channels):
        env = envelopes[:, ch]
        threshold = 0.5 * np.max(env)
        above = np.where(env > threshold)[0]
        onset_times.append(above[0] if len(above) > 0 else 50)

    print(f"\n{Path(csv_path).stem}")
    print(f"Peak samples:  {peak_times}  (range: {peak_times.max() - peak_times.min()})")
    print(f"Onset samples: {onset_times} (range: {max(onset_times) - min(onset_times)})")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    colors = ['b', 'g', 'r', 'orange']

    for ch in range(n_channels):
        axes[0].plot(window[:, ch], color=colors[ch], alpha=0.5, label=f'Ch{ch + 1} raw')
        axes[1].plot(envelopes[:, ch], color=colors[ch], label=f'Ch{ch + 1} env')
        axes[1].axvline(peak_times[ch], color=colors[ch], linestyle='--', alpha=0.7)
        axes[1].axvline(onset_times[ch], color=colors[ch], linestyle=':', alpha=0.7)

    axes[0].set_title(f'{Path(csv_path).stem} - Raw EMG')
    axes[1].set_title('Envelopes (dashed=peak, dotted=onset)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'diag_{Path(csv_path).stem}.png')
    plt.close()


# Run on one example of each swipe
from pathlib import Path

data_dir = Path('data/emg_data')

for gesture in ['SWIPE_UP', 'SWIPE_DOWN', 'SWIPE_LEFT', 'SWIPE_RIGHT']:
    files = list(data_dir.glob(f'{gesture}_*.csv'))
    if files:
        diagnose_envelope_timing(str(files[0]))
