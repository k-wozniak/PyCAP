from PyCAP.interferenceSources.interference_source import InterferenceSource

import numpy as np

# TODO Add possibility to specify SNR

class BrownianNoise(InterferenceSource):
    """ BrownianNoise function generates a Brownian noise based 
        on the inputted signal and SNR """

    def __init__(self, generation_range: tuple, final_range: tuple):
        """ generation_range (min, max) - affects how the noise is generated
            final_range (min, max) - affects the final range """ 
        self.g_range = generation_range
        self.f_range = final_range

    def add_interference(self, signal:  np.ndarray) -> np.ndarray:
        """ Return signal with added Brownian noise interference """
        noise_signal = self.generate_noise_signal(len(signal))
        signal = (signal + noise_signal)

        return signal

    def generate_noise_signal(self, length: int) -> np.ndarray:
        # Generate random numbers between min and max passed range
        signal = np.random.uniform(self.g_range[0], self.g_range[1], length)
        # Find the cumulative sum
        signal  = np.cumsum(signal)
        # shift the mean to 0
        signal = signal - np.mean(signal)
        # Normalise signal
        signal = self.normalise(signal, self.f_range[0], self.f_range[1])

        return signal

    def normalise(self, signal: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
        """ Takes signal and returns a normalised version where each value is 
            scaled between the new max and min. """
        o_min = np.min(signal) # old max
        o_max = np.max(signal) # old max

        if o_min == o_max:
            raise ValueError # Should not be equal

        signal = (new_max - new_min)/(o_max - o_min)*(signal-o_max) + new_max
        
        return signal