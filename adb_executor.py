# adb_executor.py
"""
ADB command executor for Wave gesture system.
Maps semantic intents to physical phone actions.
"""

import subprocess
import time


# Screen dimensions: 1080x2340
# Swipe coordinates calculated at 80% of screen bounds
ADB_COMMANDS = {
    "SELECT": ["adb", "shell", "input", "tap", "540", "1170"],
    "SCROLL_UP": ["adb", "shell", "input", "swipe", "540", "1755", "540", "585", "300"],
    "SCROLL_DOWN": ["adb", "shell", "input", "swipe", "540", "585", "540", "1755", "300"],
    "BACK": ["adb", "shell", "input", "swipe", "864", "1170", "216", "1170", "300"],
    "FORWARD": ["adb", "shell", "input", "swipe", "216", "1170", "864", "1170", "300"],
}

# Minimum time between commands in seconds
# Prevents command flooding if gesture fires repeatedly
COOLDOWN_SEC = 1.0


class ADBExecutor:
    def __init__(self, cooldown_sec=COOLDOWN_SEC, enabled=True):
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled
        self.last_execution_time = 0.0
        self.last_intent = None

        if self.enabled:
            if not self._verify_connection():
                print("  WARNING: No ADB device found. Commands will be skipped.")
                self.enabled = False
            else:
                print("  ADB executor ready.")

    def _verify_connection(self):
        """Check that an ADB device is connected."""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            # Filter to lines with actual devices (not the header)
            devices = [l for l in lines[1:] if '\tdevice' in l]
            return len(devices) > 0
        except Exception as e:
            print(f"  ADB connection check failed: {e}")
            return False

    def execute(self, intent: str) -> bool:
        """
        Execute an ADB command for the given intent.
        Returns True if command was executed, False if skipped.

        Args:
            intent: Semantic intent string e.g. "SELECT", "SCROLL_UP"
        """
        if not self.enabled:
            return False

        if intent not in ADB_COMMANDS:
            print(f"  [ADB] Unknown intent: {intent}")
            return False

        # Cooldown check
        now = time.time()
        elapsed = now - self.last_execution_time
        if elapsed < self.cooldown_sec:
            print(f"  [ADB] Cooldown active ({elapsed:.2f}s < {self.cooldown_sec}s), skipping {intent}")
            return False

        # Execute
        try:
            cmd = ADB_COMMANDS[intent]
            subprocess.run(cmd, capture_output=True, timeout=3)
            self.last_execution_time = time.time()
            self.last_intent = intent
            print(f"  [ADB] Executed: {intent}")
            return True
        except subprocess.TimeoutExpired:
            print(f"  [ADB] Timeout executing: {intent}")
            return False
        except Exception as e:
            print(f"  [ADB] Error executing {intent}: {e}")
            return False

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True