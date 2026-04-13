# realtime_wave/realtime_pipeline.py
"""
Live inference pipeline with proper filtered calibration and verbose debugging.
"""

import time
import joblib
import numpy as np
from pathlib import Path

from config_wave.invariants import config
from realtime_wave.ble_reader import BLEReader
from realtime_wave.realtime_wave import RealTimeWave, REST_GATE_IDX
from realtime_wave.streaming_filter import StreamingSOSFilter
from dsp_wave.filters import build_sos
from adb_executor import ADBExecutor

INTENT_MAP = {
    "FIST": "SELECT",
    "SWIPE_DOWN": "SCROLL_DOWN",
    "SWIPE_LEFT": "BACK",
    "SWIPE_RIGHT": "FORWARD",
    "SWIPE_UP": "SCROLL_UP",
}


def calibrate_baseline(reader, training_mean, training_std, duration_sec=3.0, fs=200):
    """
    Collect REST baseline with filtering.
    Computes session-relative REST threshold using 2.5 standard deviations
    above the REST RMS centroid in normalized space.
    Returns: (use_mean, use_std, rest_threshold)
    """
    print(f"\n{'=' * 60}")
    print("CALIBRATION - Keep arm relaxed and still")
    print(f"{'=' * 60}")

    sos = build_sos(fs=fs)
    cal_filter = StreamingSOSFilter(sos)

    samples_needed = int(duration_sec * fs)
    filtered_collected = []
    chunk_rms_values = []

    reader.flush_buffer()
    time.sleep(0.1)

    print(f"Collecting {duration_sec}s of REST baseline...")

    while len(filtered_collected) < samples_needed:
        chunk = reader.read_chunk()
        if chunk is not None and len(chunk) > 0:
            filtered_chunk = cal_filter.process(chunk)
            filtered_collected.extend(filtered_chunk.tolist())

            normalized_chunk = (filtered_chunk - training_mean) / (training_std + 1e-8)
            rms_per_channel = np.sqrt(np.mean(normalized_chunk ** 2, axis=0))
            chunk_rms_values.append(float(np.mean(rms_per_channel)))
        else:
            time.sleep(0.01)

        pct = min(100, len(filtered_collected) / samples_needed * 100)
        print(f"\r  Progress: {pct:.0f}%", end="", flush=True)

    print(f"\r  Collected {len(filtered_collected)} samples")

    filtered_data = np.array(filtered_collected[:samples_needed])
    filtered_mean = np.mean(filtered_data, axis=0)
    filtered_std = np.std(filtered_data, axis=0)

    print(f"\n  FILTERED signal stats:")
    print(f"    Mean: {filtered_mean}")
    print(f"    Std:  {filtered_std}")

    rms_array = np.array(chunk_rms_values)
    trim_pct = np.percentile(rms_array, 95)
    trimmed_rms = rms_array[rms_array < trim_pct]

    rest_centroid = float(np.mean(trimmed_rms))
    rest_rms_std = float(np.std(trimmed_rms))
    rest_threshold = rest_centroid + 2.5 * rest_rms_std

    print(f"\n  REST RMS distribution (normalized space):")
    print(f"    Centroid: {rest_centroid:.4f}")
    print(f"    Std:      {rest_rms_std:.4f}")
    print(f"    Threshold (centroid + 2.5σ): {rest_threshold:.4f}")

    mean_diff = np.abs(filtered_mean - training_mean)

    if np.all(mean_diff < 1.0):
        print(f"\n  ✓ Means aligned - using TRAINING stats for normalization")
        return training_mean, training_std, rest_threshold
    else:
        print(f"\n  ⚠ Mean mismatch - using live calibration stats")
        return filtered_mean, filtered_std, rest_threshold


def run(source="BLE", file_path=None, threshold=0.70, calibrate=True, verbose=True):
    """
    Main live inference loop.
    """
    model_path = Path("models/gesture_model.joblib")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'python main.py --mode train' first.")
        return

    print("Loading model...")
    bundle = joblib.load(model_path)

    classes = bundle['classes']
    models = bundle['models']
    global_mean = bundle['global_mean']
    global_std = bundle['global_std']
    best_model = bundle['best_model']
    cv_results = bundle['cv_results']

    print(f"  Classes: {list(classes)}")
    print(f"  Best model: {best_model} ({cv_results[best_model]:.2%})")
    print(f"  Training global_mean: {global_mean}")
    print(f"  Training global_std:  {global_std}")

    knn_model = models['lda_to_knn']
    logreg_model = models['lda_to_logreg']

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
        from realtime_wave.DataStreamer import create_reader
        reader = create_reader(reader_type='CSV', csv_path=file_path)
        calibrate = False
    else:
        print(f"ERROR: Unknown source '{source}'")
        return

    # Calibration
    rest_threshold = None
    if calibrate:
        use_mean, use_std, rest_threshold = calibrate_baseline(
            reader,
            training_mean=global_mean,
            training_std=global_std
        )
    else:
        use_mean = global_mean
        use_std = global_std

    print("\nInitializing inference engine...")
    processor = RealTimeWave(
        knn_model=knn_model,
        logreg_model=logreg_model,
        global_mean=use_mean,
        global_std=use_std,
        cogency_threshold=threshold,
        rest_threshold=rest_threshold
    )

    print("\nInitializing ADB executor...")
    adb = ADBExecutor(cooldown_sec=1.0, enabled=True)

    print("Flushing buffer...")
    reader.flush_buffer()

    print("\n" + "=" * 60)
    print("LIVE INFERENCE - Press Ctrl+C to stop")
    print("=" * 60)
    print(f"Cogency threshold: {threshold}")
    print(f"REST gate threshold: {rest_threshold:.4f}" if rest_threshold else "REST gate: disabled")
    print(f"Verbose: {verbose}")
    print("-" * 60)

    if verbose:
        print(f"{'Window':<8} {'Raw Pred':<12} {'Conf':<8} {'Status':<10}")
        print("-" * 60)
    else:
        print(f"{'Gesture':<15} {'Intent':<15} {'Conf':<8}")
        print("-" * 60)

    last_stable_gesture = None
    sample_counter = 0

    try:
        while True:
            chunk = reader.read_chunk()

            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue

            results = processor.process_chunk(chunk)

            if not results:
                continue

            for result in results:
                sample_counter += 1

                if isinstance(result, tuple):
                    label_idx, confidence = result
                else:
                    label_idx = result
                    confidence = 0.0

                if label_idx == REST_GATE_IDX:
                    raw_gesture = "REST"
                elif label_idx == -1:
                    raw_gesture = "UNCERTAIN"
                else:
                    raw_gesture = classes[label_idx]

                if verbose and sample_counter % 5 == 0:
                    if label_idx in (REST_GATE_IDX, -1):
                        status = "HOLD"
                    elif last_stable_gesture is None:
                        status = "FIRST"
                    elif raw_gesture != last_stable_gesture:
                        status = "CHANGE"
                    else:
                        status = "same"
                    print(f"  [{sample_counter:4d}]  {raw_gesture:<12} {confidence:5.0%}    ({status})")

                if label_idx in (REST_GATE_IDX, -1):
                    if label_idx == REST_GATE_IDX:
                        last_stable_gesture = None
                    continue

                gesture = classes[label_idx]
                intent = INTENT_MAP.get(gesture, "UNKNOWN")

                if gesture != last_stable_gesture:
                    if verbose:
                        print(f">>> CONFIRMED: {gesture:<12} {intent:<12} {confidence:<6.0%} <<<")
                    else:
                        print(f"{gesture:<15} {intent:<15} {confidence:<8.0%}")
                    last_stable_gesture = gesture
                    adb.execute(intent)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(f"Session Summary:")
        print(f"  Total windows processed: {sample_counter}")
        print(f"  Duration: ~{sample_counter * config.hop / config.fs:.1f}s")
        print(f"  Final gesture: {last_stable_gesture}")
        print("=" * 60)

    finally:
        if reader is not None:
            reader.close()
        print("Done.")


if __name__ == "__main__":
    run()