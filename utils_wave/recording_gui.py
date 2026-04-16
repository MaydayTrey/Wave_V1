"""
Tkinter-based GUI for recording EMG gesture data.

Usage:
    python -m utils_wave.recording_gui --gesture FIST --trials 15
    python -m utils_wave.recording_gui --gesture REST --trials 10
    python -m utils_wave.recording_gui --simulate  # Test without hardware

Controls:
    SPACE - Start recording
    ESC   - Stop current recording
    Q     - Quit application
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import threading
import numpy as np
import argparse
import time

try:
    import winsound

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

from config_wave import config


@dataclass
class GestureProtocol:
    """Timing protocol for a gesture recording."""
    name: str
    hold_time: float
    rest_before: float
    rest_after: float
    description: str


GESTURE_PROTOCOLS = {
    "REST": GestureProtocol(
        name="REST",
        hold_time=3.0,
        rest_before=0.0,
        rest_after=0.0,
        description="Relax your hand completely"
    ),
    "FIST": GestureProtocol(
        name="FIST",
        hold_time=1.0,
        rest_before=1.0,
        rest_after=1.0,
        description="Make a firm fist"
    ),
    "SWIPE_UP": GestureProtocol(
        name="SWIPE_UP",
        hold_time=1.0,
        rest_before=1.0,
        rest_after=1.0,
        description="Extend fingers upward quickly"
    ),
    "SWIPE_DOWN": GestureProtocol(
        name="SWIPE_DOWN",
        hold_time=1.0,
        rest_before=1.0,
        rest_after=1.0,
        description="Flex fingers downward quickly"
    ),
    "SWIPE_LEFT": GestureProtocol(
        name="SWIPE_LEFT",
        hold_time=1.0,
        rest_before=1.0,
        rest_after=1.0,
        description="Deviate wrist left (thumb side)"
    ),
    "SWIPE_RIGHT": GestureProtocol(
        name="SWIPE_RIGHT",
        hold_time=1.0,
        rest_before=1.0,
        rest_after=1.0,
        description="Deviate wrist right (pinky side)"
    ),
}


class SimulatedReader:
    """Simulated EMG reader for testing without hardware."""

    def __init__(self, chunk_size=config.hop):
        self.chunk_size = chunk_size
        self.fs = config.fs

    def flush_buffer(self):
        print("Buffer flushed (simulated)")

    def read_chunk(self):
        time.sleep(self.chunk_size / self.fs)
        return np.random.randn(self.chunk_size, 4) * 100

    def close(self):
        pass


class RecordingGUI:
    """GUI for recording EMG gesture data."""

    def __init__(self, reader=None, gesture: str = None, trials: int = 15):
        self.reader = reader
        self.recording = False
        self.current_data = []
        self.target_trials = trials
        self.preset_gesture = gesture

        self.rep_counts = {g: 0 for g in GESTURE_PROTOCOLS}

        self.root = tk.Tk()
        self.root.title("EMG Gesture Recorder")
        self.root.geometry("650x580")
        self.root.resizable(False, False)

        # Bind keyboard shortcuts
        self.root.bind("<space>", lambda e: self._on_space())
        self.root.bind("<Escape>", lambda e: self._stop_recording())
        self.root.bind("q", lambda e: self._quit())
        self.root.bind("Q", lambda e: self._quit())

        self._build_ui()

        if self.preset_gesture:
            self.gesture_var.set(self.preset_gesture)
            self._update_protocol_display()
            self._lock_gesture_selection()

    def _build_ui(self):
        """Build the GUI components."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="EMG Gesture Recorder",
            font=("Helvetica", 18, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Gesture selection frame
        select_frame = ttk.LabelFrame(main_frame, text="Gesture Selection", padding=10)
        select_frame.pack(fill="x", pady=5)

        self.gesture_var = tk.StringVar(value="FIST")

        dropdown_frame = ttk.Frame(select_frame)
        dropdown_frame.pack(fill="x")

        ttk.Label(dropdown_frame, text="Select Gesture:").pack(side="left", padx=(0, 10))

        gesture_menu = ttk.Combobox(
            dropdown_frame,
            textvariable=self.gesture_var,
            values=list(GESTURE_PROTOCOLS.keys()),
            state="readonly",
            width=20
        )
        gesture_menu.pack(side="left")
        gesture_menu.bind("<<ComboboxSelected>>", lambda e: self._update_protocol_display())
        self.gesture_menu = gesture_menu

        # Protocol info display
        info_frame = ttk.LabelFrame(main_frame, text="Protocol Info", padding=10)
        info_frame.pack(fill="x", pady=5)

        self.protocol_label = ttk.Label(
            info_frame,
            text="",
            wraplength=550,
            font=("Helvetica", 10)
        )
        self.protocol_label.pack()

        # Trials progress frame
        trials_frame = ttk.LabelFrame(main_frame, text="Trials Progress", padding=10)
        trials_frame.pack(fill="x", pady=5)

        self.trials_label = ttk.Label(
            trials_frame,
            text="Completed: 0 / 0",
            font=("Helvetica", 11)
        )
        self.trials_label.pack()

        self.trials_bar = ttk.Progressbar(
            trials_frame,
            length=550,
            mode="determinate"
        )
        self.trials_bar.pack(pady=5, fill="x")

        # Current recording status frame
        status_frame = ttk.LabelFrame(main_frame, text="Current Recording", padding=15)
        status_frame.pack(fill="both", expand=True, pady=5)

        # Phase label (REST / GESTURE NAME)
        self.phase_label = tk.Label(
            status_frame,
            text="READY",
            font=("Helvetica", 42, "bold"),
            fg="gray",
            pady=10
        )
        self.phase_label.pack(pady=(5, 10))

        # Phase progress bar
        self.phase_bar = ttk.Progressbar(
            status_frame,
            length=500,
            mode="determinate",
            maximum=100
        )
        self.phase_bar.pack(pady=5)

        # Time remaining label
        self.time_label = ttk.Label(
            status_frame,
            text="",
            font=("Helvetica", 14)
        )
        self.time_label.pack(pady=5)

        # Instruction label
        self.instruction_label = ttk.Label(
            status_frame,
            text="Press SPACE to start recording",
            wraplength=500,
            font=("Helvetica", 11)
        )
        self.instruction_label.pack(pady=(10, 5))

        # Keyboard hints
        hints_frame = ttk.Frame(main_frame)
        hints_frame.pack(fill="x", pady=(10, 0))

        hints_text = "SPACE = Start Recording  |  ESC = Stop  |  Q = Quit"
        ttk.Label(
            hints_frame,
            text=hints_text,
            font=("Helvetica", 10, "italic"),
            foreground="gray"
        ).pack()

        self._update_protocol_display()

    def _lock_gesture_selection(self):
        self.gesture_menu.configure(state="disabled")

    def _update_protocol_display(self):
        gesture = self.gesture_var.get()
        protocol = GESTURE_PROTOCOLS[gesture]

        total_time = protocol.rest_before + protocol.hold_time + protocol.rest_after

        info_text = (
            f"Gesture: {protocol.name}  |  {protocol.description}\n"
            f"Timeline: REST ({protocol.rest_before}s) → GESTURE ({protocol.hold_time}s) → REST ({protocol.rest_after}s)  |  Total: {total_time}s"
        )
        self.protocol_label.config(text=info_text)

        self.trials_bar["maximum"] = self.target_trials
        self.trials_bar["value"] = self.rep_counts[gesture]
        self.trials_label.config(
            text=f"Completed: {self.rep_counts[gesture]} / {self.target_trials}"
        )

    def _set_phase(self, text: str, color: str):
        """Update phase label."""
        self.phase_label.config(text=text, fg=color)

    def _set_phase_bar(self, value: float):
        """Update phase progress bar (0-100)."""
        self.phase_bar["value"] = value

    def _set_time(self, text: str):
        """Update time label."""
        self.time_label.config(text=text)

    def _set_instruction(self, text: str):
        """Update instruction label."""
        self.instruction_label.config(text=text)

    def _play_beep(self, frequency: int = 440, duration: int = 200):
        if HAS_AUDIO:
            try:
                winsound.Beep(frequency, duration)
            except:
                pass

    def _on_space(self):
        if not self.recording:
            self._start_recording()

    def _start_recording(self):
        if self.recording:
            return
        thread = threading.Thread(target=self._recording_thread, daemon=True)
        thread.start()

    def _run_phase(self, phase_name: str, duration: float, color: str,
                   instruction: str, record_data: bool = False):
        """
        Run a timed phase with progress bar animation.

        Args:
            phase_name: Text to display (e.g., "REST", "FIST")
            duration: Duration in seconds
            color: Color for phase label
            instruction: Instruction text
            record_data: Whether to record EMG during this phase
        """
        if duration <= 0:
            return

        # Set up phase display
        self.root.after(0, lambda: self._set_phase(phase_name, color))
        self.root.after(0, lambda: self._set_instruction(instruction))
        self.root.after(0, lambda: self._set_phase_bar(0))

        # Play start beep
        if phase_name != "REST":
            self._play_beep(880, 200)
        else:
            self._play_beep(440, 100)

        # Run phase with progress updates
        update_interval = 0.05  # 50ms updates for smooth animation
        start_time = time.time()

        while self.recording:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # Update progress bar
            progress = (elapsed / duration) * 100
            remaining = duration - elapsed

            self.root.after(0, lambda p=progress: self._set_phase_bar(p))
            self.root.after(0, lambda r=remaining: self._set_time(f"{r:.1f}s remaining"))

            # Record data if requested
            if record_data and self.reader is not None:
                chunk = self.reader.read_chunk()
                if chunk.size > 0:
                    self.current_data.append(chunk)
            else:
                time.sleep(update_interval)

        # Fill bar to 100% at end
        self.root.after(0, lambda: self._set_phase_bar(100))
        self.root.after(0, lambda: self._set_time(""))

    def _recording_thread(self):
        """Recording sequence in background thread."""
        gesture = self.gesture_var.get()
        protocol = GESTURE_PROTOCOLS[gesture]

        # Flush stale buffer
        if self.reader is not None:
            self.reader.flush_buffer()

        self.recording = True
        self.current_data = []

        try:
            # Countdown: 3-2-1
            for count in [3, 2, 1]:
                if not self.recording:
                    return
                self.root.after(0, lambda c=count: self._set_phase(str(c), "orange"))
                self.root.after(0, lambda: self._set_instruction("Get ready..."))
                self.root.after(0, lambda: self._set_phase_bar(0))
                self.root.after(0, lambda: self._set_time(""))
                self._play_beep(440, 150)
                time.sleep(1.0)

            if not self.recording:
                return

            # Flush buffer right before recording sequence
            if self.reader is not None:
                self.reader.flush_buffer()

            # Phase 1: Pre-gesture REST
            if protocol.rest_before > 0:
                self._run_phase(
                    phase_name="REST",
                    duration=protocol.rest_before,
                    color="blue",
                    instruction="Relax your hand...",
                    record_data=False
                )

            if not self.recording:
                return

            # Flush buffer right before gesture recording
            if self.reader is not None:
                self.reader.flush_buffer()

            # Phase 2: GESTURE (record data)
            self._run_phase(
                phase_name=protocol.name,
                duration=protocol.hold_time,
                color="green",
                instruction=protocol.description,
                record_data=True
            )

            if not self.recording:
                return

            # Phase 3: Post-gesture REST
            if protocol.rest_after > 0:
                self._run_phase(
                    phase_name="REST",
                    duration=protocol.rest_after,
                    color="blue",
                    instruction="Relax your hand...",
                    record_data=False
                )

            # Done
            self._play_beep(660, 400)
            self.root.after(0, lambda: self._set_phase("DONE", "gray"))
            self.root.after(0, lambda: self._set_phase_bar(100))
            self.root.after(0, lambda: self._set_time(""))

            # Save recording
            if self.recording and len(self.current_data) > 0:
                self._save_recording(gesture)

        except Exception as e:
            self.root.after(0, lambda: self._set_phase("ERROR", "red"))
            self.root.after(0, lambda err=str(e): self._set_instruction(err))
            print(f"Recording error: {e}")

        finally:
            self.recording = False
            self.root.after(0, self._recording_complete)

    def _save_recording(self, gesture: str):
        """Save recorded data to CSV."""
        data = np.vstack(self.current_data)
        n_samples = len(data)

        timestamps = np.arange(n_samples) / config.fs

        output_dir = Path("training_data") / gesture
        output_dir.mkdir(parents=True, exist_ok=True)

        self.rep_counts[gesture] += 1
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{timestamp_str}_rep{self.rep_counts[gesture]:03d}.csv"
        filepath = output_dir / filename

        output_data = np.column_stack([timestamps, data])
        header = "timestamp,ch1,ch2,ch3,ch4"
        np.savetxt(filepath, output_data, delimiter=",", header=header, comments="")

        print(f"Saved {n_samples} samples to {filepath}")

    def _recording_complete(self):
        """Called on main thread when recording finishes."""
        self._update_protocol_display()

        gesture = self.gesture_var.get()
        if self.rep_counts[gesture] >= self.target_trials:
            self._set_instruction(
                f"Target reached! {self.rep_counts[gesture]} recordings completed."
            )
            messagebox.showinfo(
                "Complete",
                f"Completed {self.target_trials} recordings for {gesture}!"
            )
        else:
            remaining = self.target_trials - self.rep_counts[gesture]
            self._set_instruction(
                f"Recording saved. {remaining} more to go. Press SPACE to continue."
            )

    def _stop_recording(self):
        if self.recording:
            self.recording = False
            self.root.after(0, lambda: self._set_phase("STOPPED", "red"))
            self.root.after(0, lambda: self._set_instruction("Recording stopped. Press SPACE to try again."))

    def _quit(self):
        self.recording = False
        if self.reader is not None:
            self.reader.close()
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="EMG Gesture Recording GUI")
    parser.add_argument(
        "--gesture",
        type=str,
        choices=list(GESTURE_PROTOCOLS.keys()),
        help="Pre-select gesture (locks selection)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Target number of trials to record (default: 15)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated data (no hardware required)"
    )

    args = parser.parse_args()

    if args.simulate:
        print("Using simulated EMG data")
        reader = SimulatedReader()
    else:
        try:
            from realtime_wave import BLEReader
            reader = BLEReader()
        except Exception as e:
            print(f"Failed to connect to hardware: {e}")
            print("Falling back to simulation mode")
            reader = SimulatedReader()

    gui = RecordingGUI(reader=reader, gesture=args.gesture, trials=args.trials)
    gui.run()


if __name__ == "__main__":
    main()
