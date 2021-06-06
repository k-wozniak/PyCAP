import numpy as np
from typing import Dict

class RecordedSignals:
    key_round: int = 6

    fs: int # Sampling Frequency Hz

    """ Storage of the signlas where
        Key: distance from the excitation source and the electrode rounded to 
        6 places so 0.08484848 would be 0.084848 using round(a, 6)
        Value: one dimensional array with the signal
    """
    data: Dict[float, np.ndarray] = None

    def __init__(self, fs, distances_list, signals_list):
        # Check if len of distances and signals is the same
        if len(distances_list) is not len(signals_list):
            raise Exception("Number of electrode distances is different to the number of signals.")
        
        # Check if each signals has the same length
        l = -1
        for s in signals_list:
            if l == -1:
                l = len(s)
            elif l != len(s):
                raise Exception("Not every signal is the same length.")

        self.fs = fs

        self.data = {}
        for i in range(len(distances_list)):
            self.data[round(distances_list[i], self.key_round)] = signals_list[i]

    def add_signal(self, location: float, signal: np.ndarray):
        self.data[round(location, self.key_round)] = signal

    def remove_signal(self, location: float) -> bool:
        location = round(location, self.key_round)

        if location in self.data:
            self.data.pop(location)
            return True
        else:
            return False