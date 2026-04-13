"""
Real-time EMG processing pipeline.

This module provides streaming EMG processing components including:
- BLE data streaming from Ganglion board
- Real-time filtering and normalization
- Rolling buffer for windowed feature extraction
- Cogency-based prediction fusion
- Temporal persistence for stable outputs
"""

from .ble_reader import BLEReader
from .streaming_filter import StreamingSOSFilter
from .rolling_buffer import RollingBuffer
from .cogency import cogency_fusion
from .persistence import temporal_persistence
from .realtime_wave import RealTimeWave

__all__ = [
    "BLEReader",
    "StreamingSOSFilter",
    "RollingBuffer",
    "cogency_fusion",
    "temporal_persistence",
    "RealTimeWave",
]
