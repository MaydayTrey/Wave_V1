import serial
import numpy as np
import time
from config_wave import config

class SerialReader:
    """Reads EMG samples from Arduino serial output.
    Returns chunks of samples sized for the realtime pipeline."""
# Make the serial COM Port an optional use, with BLE as an option as well.

    def __init__(self, port='COM3', baudrate=115200, chunk_size=config.hop):
        self.port = port
        self.baudrate = baudrate
        self.chunk_size = chunk_size

        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f"Connected to {self.port} at {self.baudrate}")
    def read_chunk(self):
        samples = []
        while len(samples) < self.chunk_size:
            line = self.ser.readline().decode("utf-8").strip()

            if not line:
                continue
            try:
                values = [float(v) for v in line.split(",")]
                if len(values) != 4:
                    continue
                samples.append(values)
            except ValueError:
                continue
        return np.array(samples, dtype=float)

