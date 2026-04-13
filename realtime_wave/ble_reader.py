from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BoardShim.release_all_sessions()
from config_wave import config
import numpy as np
import time


class BLEReader:

    def __init__(self, chunk_size=config.hop):
        params = BrainFlowInputParams()
        BoardShim.enable_dev_board_logger()
        params.mac_address = 'F4:11:52:42:04:E0'
        params.serial_port = "COM3"
        params.timeout = 100
        self.board = BoardShim(BoardIds.GANGLION_BOARD, params)
        self.chunk_size = chunk_size

        # Rate limiting
        self.fs = config.fs  # 200 Hz
        self.chunk_interval = chunk_size / self.fs  # 50/200 = 0.25s
        self.last_read_time = None

        print("\nPreparing session....")
        for attempt in range(2):
            try:
                print(f"Preparing Session attempt {attempt + 1}")
                self.board.prepare_session()
                self.board.config_board("x")  # Disable accelerometer
                time.sleep(3)
                print("\nInitializing reader...")
                self.board.start_stream()
                print("\nConnected successfully")
                break
            except Exception as e:
                print(e)
                print("Failed Connection, retrying.")
                BoardShim.release_all_sessions()
                time.sleep(3)

    def flush_buffer(self):
        """Clear any accumulated data in the BrainFlow buffer."""
        self.board.get_board_data()  # Pulls and discards all buffered data
        self.last_read_time = None   # Reset rate limiting
        print("Buffer flushed")

    def read_chunk(self):
        """Read a chunk of EMG data with rate limiting."""
        # Rate limit: wait until enough time has passed for new samples
        now = time.time()
        if self.last_read_time is not None:
            elapsed = now - self.last_read_time
            wait_time = self.chunk_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        self.last_read_time = time.time()

        # Get all available samples (clears buffer)
        data = self.board.get_board_data()

        if data.size == 0:
            return np.empty((0, 4))

        emg_channels = BoardShim.get_emg_channels(BoardIds.GANGLION_BOARD)
        emg = data[emg_channels, :].T  # Shape: (samples, 4)

        return np.array(emg, dtype=float)

    def close(self):
        """Clean up BrainFlow session."""
        try:
            print("\nStopping stream....")
            self.board.stop_stream()
        except:
            pass
        try:
            print("\nReleasing session...")
            self.board.release_session()
        except:
            pass
