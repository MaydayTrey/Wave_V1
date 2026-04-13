# realtime_wave/realtime_pipeline.py
"""
Live inference pipeline - orchestrates streaming EMG to gesture recognition.
"""

import time
import joblib
import numpy as np
from pathlib import Path

from config_wave.invariants import config
from realtime_wave.ble_reader import BLEReader
from realtime_wave.realtime_wave import RealTimeWave

# Gesture to Intent mapping
INTENT_MAP = {
    "FIST": "SELECT",
    "REST": "IDLE",
    "SWIPE_DOWN": "SCROLL_DOWN",
    "SWIPE_LEFT": "BACK",
    "SWIPE_RIGHT": "FORWARD",
    "SWIPE_UP": "SCROLL_UP",
}


def run(source="BLE", file_path=None, threshold=0.55):
    """
    Main live inference loop.

    Args:
        source: "BLE", "serial", or "CSV"
        file_path: Path to CSV file (only for source="CSV")
        threshold: Cogency confidence threshold
    """

    model_path = Path("models/gesture_model.joblib")

    # Load model bundle
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'python main.py --mode train' first.")
        return

    print("Loading model...")
    bundle = joblib.load(model_path)

    # Extract components from bundle
    classes = bundle['classes']
    models = bundle['models']
    global_mean = bundle['global_mean']
    global_std = bundle['global_std']
    best_model = bundle['best_model']
    cv_results = bundle['cv_results']

    print(f"  Classes: {classes}")
    print(f"  Best model: {best_model} ({cv_results[best_model]:.2%})")
    print(f"  Global mean: {global_mean}")
    print(f"  Global std: {global_std}")

    # Get the two models for cogency fusion
    knn_model = models['lda_to_knn']
    logreg_model = models['lda_to_logreg']

    # Initialize data source
    reader = None

    if source == "BLE":
        print("\nConnecting to Ganglion via BLE...")
        try:
            reader = BLEReader(chunk_size=config.hop)
            print("Pipeline ready!")
        except Exception as e:
            print(f"ERROR: Failed to connect to Ganglion: {e}")
            return

    elif source == "CSV":
        if file_path is None:
            print("ERROR: --file required for CSV source")
            return
        print(f"Replaying from {file_path}...")
        raise NotImplementedError("CSV replay not yet implemented")

    else:
        print(f"ERROR: Unknown source '{source}'")
        return

    # Initialize real-time processor
    print("Initializing inference engine...")
    processor = RealTimeWave(
        knn_model=knn_model,
        logreg_model=logreg_model,
        global_mean=global_mean,
        global_std=global_std,
        cogency_threshold=threshold
    )

    # Flush any stale data from buffer
    print("Flushing buffer...")
    reader.flush_buffer()

    print("\n" + "=" * 60)
    print("LIVE INFERENCE - Press Ctrl+C to stop")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Classes: {list(classes)}")
    print("-" * 60)
    print(f"{'Gesture':<15} {'Intent':<15} {'Confidence':<12}")
    print("-" * 60)

    last_gesture = None
    last_print_time = 0

    try:
        while True:
            # Read chunk from source
            chunk = reader.read_chunk()

            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue

            # Process through pipeline
            results = processor.process_chunk(chunk)

            if not results:
                continue

            for result in results:
                current_time = time.time()

                # Handle different return formats
                if isinstance(result, tuple):
                    label_idx, confidence = result
                else:
                    label_idx = result
                    confidence = 0.0

                if label_idx == -1:
                    gesture = "UNCERTAIN"
                    intent = "---"
                else:
                    gesture = classes[label_idx]
                    intent = INTENT_MAP.get(gesture, "UNKNOWN")

                # Print on gesture change, or every 0.5s for feedback
                should_print = (
                        gesture != last_gesture or
                        (current_time - last_print_time) > 0.5
                )

                if should_print and gesture != "UNCERTAIN":
                    print(f"{gesture:<15} {intent:<15} {confidence:<12.2%}")
                    last_gesture = gesture
                    last_print_time = current_time

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopping...")

    finally:
        if reader is not None:
            reader.close()
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    run()
