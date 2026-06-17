import numpy as np
from scipy.signal import butter, lfilter


class EEGPreprocessor:
    def __init__(self, fs=250):
        self.fs = fs

    def bandpass(self, data, low=1, high=40, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return lfilter(b, a, data, axis=0)

    def preprocess(self, data):
        return self.bandpass(data)
