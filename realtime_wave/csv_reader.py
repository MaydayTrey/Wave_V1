#from config_wave import config
import numpy as np
#import joblib
import pandas as pd
from config_wave import config

'''We must take in the CSV that already exists, and we essentially would likely just use
joblib to load it, and parse it by chunks of some size, and send it to this which would pass it into serial reader..
It must return read chunk so it must behave almost like a sliding window in itself,
it pushes up by the index value, likely using pandas dataframe which automatically has that nice index 
feature. '''

'''Dont use joblib to load CSV
Store the data as NumPy Array, not DataFrame --> Faster for streaming
Maintain an internal cursor, so each read chunk returns the next slice
read_chunk should not take chunk size as a parameter. 
Handle end of file behavior, stop, pad, or loop. '''

class CSVReader:
    def __init__(self, csv_path, chunk_size=config.chunk_size, loop=True):
        self.chunk_size = chunk_size
        self.loop = loop
        df = pd.read_csv(csv_path)
        data = df.to_numpy(dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        self.data = data
        self.cursor = 0
        self.total_samples = data.shape[0]
        print(f"CSVReader loaded {self.total_samples} samples, with shape of: {data.shape[1]}" )
        #Then turn the loaded CSV into pandas DataFrame
        #Then we can use the internal indice of the DataFrame to slide down the EMG data!
    def read_chunk(self):
        start = self.cursor
        end = start + self.chunk_size

        #Normal case
        if end <= self.total_samples:
            chunk = self.data[start:end]
        else:
            chunk = self.data[start:self.total_samples]
            if self.loop:
                remaining = self.chunk_size - chunk.shape[0]
                chunk2 = self.data[:remaining]
                chunk = np.vstack((chunk, chunk2))
                end = remaining
            else:
                pass
        self.cursor = end % self.total_samples
        return np.asarray(chunk, dtype=float)