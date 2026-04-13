import numpy as np
import pandas as pd
import time
from pathlib import Path
from config_wave import config
from realtime_wave.DataStreamer import create_reader

def record_sets(reader, save_dir="GestureCSVs"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seconds = int(input("\nEnter recording time per gesture (seconds): "))
    label_quantity = int(input("How many labels? "))
    recording_quantity = int(input("How many recordings per label? "))
    username = input("Enter username: ")
    try:
        for x in range(label_quantity):
            label_name = input("\nEnter label name: ")

            label_dir = save_dir / label_name
            label_dir.mkdir(exist_ok=True)

            for i in range(recording_quantity):
                print(f"\nPrepare for {label_name}, recording {i+1}/{recording_quantity}")
                input("Press ENTER to start...")
                reader.read_chunk()
                start = time.time()
                emg_data = []
                target_samples = int(config.fs*seconds)
                collected_samples = 0


                while collected_samples < target_samples:
                    chunk = reader.read_chunk()

                    if chunk is not None and chunk.shape[0] > 0:
                        if chunk.shape[1] == config.n_channels:
                            emg_data.append(chunk)
                            collected_samples += chunk.shape[0]
                        else:
                            print("Chunk size is incorrect.")
                            print("Chunk shape: ", chunk.shape)
                    time.sleep(config.hop / config.fs)

                if len(emg_data) == 0:
                    print("No valid data collected, skipping.")
                    print("Chunk shape: ", chunk.shape)
                    continue

                data = np.vstack(emg_data)

                # Trim startup transient
                if data.shape[0] > 40:
                    data = data[40:]

                timestamp = int(time.time())
                filename = f"{label_name}_{username}_rep{i+1}_{timestamp}.csv"
                filepath = label_dir / filename

                pd.DataFrame(data).to_csv(filepath, index=False)

                print(f"Saved: {filepath}")
                print(f"Shape: {data.shape}")
        #After recording for all labels and saving, release the board, and stop
    finally:
        print("Finished. Releasing board.")
        reader.close()


if __name__ == "__main__":
    reader_type = input("\n Please enter the reader type: serial, BLE, or CSV. ").strip()
    if reader_type == 'CSV':
        path = input("\nEnter path to CSV file: ")
        reader = create_reader(reader_type='CSV', csv_path=path)
    else:
        reader = create_reader(reader_type=reader_type)
    record_sets(reader)