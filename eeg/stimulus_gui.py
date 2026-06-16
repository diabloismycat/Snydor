"""
Snydor EEG Stimulus GUI
Motor Imagery Experimental Interface

Used to generate labeled cues for EEG recording:
REST vs MOTOR IMAGERY
"""

import time
import csv

try:
    import tkinter as tk
except ImportError:
    tk = None


class StimulusGUI:
    def __init__(self, trial_time=4):
        self.trial_time = trial_time
        self.log = []

    def record_event(self, label):
        self.log.append([time.time(), label])

    def save(self, filename="stimulus_log.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label"])
            writer.writerows(self.log)

    def run_console(self, trials=5):
        for _ in range(trials):
            print("REST")
            self.record_event("REST")
            time.sleep(self.trial_time)

            print("MI")
            self.record_event("MI")
            time.sleep(self.trial_time)

        self.save()

    def run_gui(self, trials=5):
        if tk is None:
            self.run_console(trials)
            return

        root = tk.Tk()
        root.title("Snydor EEG Stimulus")

        label = tk.Label(root, text="READY", font=("Arial", 24))
        label.pack(expand=True)

        def loop(i=0):
            if i >= trials:
                self.save()
                label.config(text="DONE")
                return

            label.config(text="REST")
            self.record_event("REST")
            root.update()
            time.sleep(self.trial_time)

            label.config(text="MI")
            self.record_event("MI")
            root.update()
            time.sleep(self.trial_time)

            root.after(100, lambda: loop(i + 1))

        root.after(1000, lambda: loop(0))
        root.mainloop()


if __name__ == "__main__":
    gui = StimulusGUI()
    gui.run_console()
