import time
import csv
import numpy as np


class EEGRecorder:
    def __init__(self, channels=4, fs=250):
        self.channels = channels
        self.fs = fs

    def read_sample(self):
        return np.random.randn(self.channels) * 1e-5

    def record(self, duration_sec, label, filename):
        samples = int(self.fs * duration_sec)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t"] + [f"ch{i}" for i in range(self.channels)] + ["label"])

            for i in range(samples):
                t = i / self.fs
                x = self.read_sample()
                writer.writerow([t] + x.tolist() + [label])
                time.sleep(1 / self.fs)

        print(f"Saved {filename} | {label}")
