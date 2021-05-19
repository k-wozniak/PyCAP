from PyCAP.interferenceSources.interference_source import InterferenceSource

import numpy as np

class GaussianWhiteNoise(InterferenceSource):
    """ GaussianWhiteNoise function generates a white gaussian noise based 
        on the inputted signal and SNR """

    def __init__(self, SNR_db: float, noise_mean: float = 0):
        self.SNR_db = SNR_db
        self.noise_mean = noise_mean

    def add_interference(self, signal:  np.ndarray) -> np.ndarray:
        """ Return signal with added gaussian white noise interference """
        noise_power = self.get_required_noise_power(signal)
        noise_signal = self.generate_noise_signal(noise_power, len(signal))

        signal = (signal + noise_signal)
        return signal

    def generate_noise_signal(self, noise_power: float, length: int) -> np.ndarray:
        return np.random.normal(self.noise_mean, np.sqrt(noise_power), length)

    def get_linear_SNR(self):
        return 10.0**(self.SNR_db/10.0)

    def get_required_noise_power(self, signal: np.ndarray) -> float:
        signal_power = np.mean((signal ** 2))
        noise_power = signal_power / self.get_linear_SNR()

        if (noise_power == 0): # In case of a null signal
            noise_power = self.get_linear_SNR()

        return noise_power