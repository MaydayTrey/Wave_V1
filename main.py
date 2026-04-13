# main.py

import argparse
import numpy as np
from pathlib import Path

from data_wave.dataset_builder import build_dataset
from ml_wave.training import train_and_save_models

# Training dataset mapping: 60 files (10 reps × 6 gestures)
TRAINING_DATA = {
    "training_data/FIST": "FIST",
    "training_data/SWIPE_DOWN": "SWIPE_DOWN",
    "training_data/SWIPE_LEFT": "SWIPE_LEFT",
    "training_data/SWIPE_RIGHT": "SWIPE_RIGHT",
    "training_data/SWIPE_UP": "SWIPE_UP",
}


def train():
    """Train gesture recognition model."""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    print("\nBuilding dataset...")

    # Collect all files with labels
    file_label_pairs = []
    for folder, label in TRAINING_DATA.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"  Warning: {folder} does not exist, skipping...")
            continue

        csv_files = sorted(folder_path.glob("*.csv"))
        print(f"  {label}: {len(csv_files)} files")

        for f in csv_files:
            file_label_pairs.append((f, label))

    if not file_label_pairs:
        print("Error: No training data found!")
        return

    # Build dataset (returns global stats now)
    X, y, groups, global_mean, global_std = build_dataset(file_label_pairs)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Global mean: {global_mean}")
    print(f"Global std: {global_std}")

    # Train models (pass stats for saving)
    train_and_save_models(X, y, groups, global_mean, global_std)

    print("\n" + "=" * 60)
    print("Training complete! Run 'python main.py --mode live' to test.")
    print("=" * 60)


def live():
    """Run live gesture recognition."""
    print("=" * 60)
    print("LIVE MODE")
    print("=" * 60)

    # Import here to avoid loading BrainFlow during training
    from realtime_wave.realtime_pipeline import run

    run(source="BLE", threshold=0.55)


def main():
    parser = argparse.ArgumentParser(description="Wave Gesture Recognition")
    parser.add_argument(
        "--mode",
        choices=["train", "live"],
        required=True,
        help="train: Train model from CSVs/, live: Real-time BLE recognition"
    )
    parser.add_argument(
        "--source",
        choices=["BLE", "serial", "CSV"],
        default="BLE",
        help="Data source for live mode (default: BLE)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="CSV file for replay (only with --source CSV)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,  # raised from 0.55
        help="Confidence threshold (default: 0.70)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "live":
        # Import here to avoid loading BrainFlow during training
        from realtime_wave.realtime_pipeline import run
        run(source=args.source, file_path=args.file, threshold=args.threshold)


if __name__ == "__main__":
    main()
