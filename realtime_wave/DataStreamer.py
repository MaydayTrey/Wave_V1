from realtime_wave.serial_reader import SerialReader
from realtime_wave.ble_reader import BLEReader
from realtime_wave.csv_reader import CSVReader
'''This type of behavior is called a FACTORY,
Given a type, return the correct reader object.

'''

def create_reader(reader_type="serial", **kwargs):
    if reader_type == "serial":
        return SerialReader(**kwargs)
    elif reader_type == "BLE":
        return BLEReader(**kwargs)
    elif reader_type == "CSV":
        return CSVReader(**kwargs)
    else:
        raise ValueError(f"Your Reader Type {reader_type} is not supported.")


'''
In our realtime_pipeline we have 
 !!! reader.read_chunk() !!!
So we must ensure each one of these has that function and it outputs the same
!!! (samples, channels) !!! as Serial does, so they're consistent across all reader types.
 with the **kwargs, we allow any number '''

