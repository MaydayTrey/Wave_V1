# realtime_wave/persistence.py

from collections import deque
import numpy as np

# Gesture classes that are ballistic (fast, brief)
# These get shorter persistence requirements
SWIPE_CLASSES = {1, 2, 3, 4}  # indices for SWIPE_DOWN, SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP
FIST_CLASS = 0

class temporal_persistence:
    """
    Requires N consecutive identical non-uncertain predictions.
    Swipe gestures use window_length=2, sustained gestures use window_length=3.
    """

    def __init__(self, window_length: int = 3, swipe_window_length: int = 2):
        self.window_length = window_length
        self.swipe_window_length = swipe_window_length
        self.history = deque(maxlen=window_length)  # max size is the larger value
        self.last_output = -1

    def _required_streak(self, label: int) -> int:
        """Return required streak length for a given label."""
        if label in SWIPE_CLASSES:
            return self.swipe_window_length
        return self.window_length

    def update(self, labels):
        """
        Process array of labels.
        Returns list of final labels (-1 if streak not achieved or same as last).
        """
        outputs = []
        for label in labels:
            self.history.append(label)

            # Need at least swipe_window_length entries to evaluate anything
            if len(self.history) < self.swipe_window_length:
                outputs.append(-1)
                continue

            hist = np.array(self.history)

            # UNCERTAIN breaks streak
            if np.any(hist == -1):
                self.last_output = -1
                outputs.append(-1)
                continue

            # Check if all match
            if not np.all(hist == hist[0]):
                outputs.append(-1)
                continue

            confirmed = hist[0]
            required = self._required_streak(confirmed)

            # Check if we have enough consecutive matching windows
            if len(self.history) < required:
                outputs.append(-1)
                continue

            # Check the last `required` entries all match
            recent = list(self.history)[-required:]
            if not all(x == confirmed for x in recent):
                outputs.append(-1)
                continue

            # Suppress retriggering
            if confirmed == self.last_output:
                outputs.append(-1)
                continue

            # Confirmed
            self.last_output = confirmed
            self.history.clear()
            outputs.append(confirmed)

        return outputs

    def reset(self):
        """Clear history and trigger lock."""
        self.history.clear()
        self.last_output = -1