import unittest
import numpy as np

from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise

class TestGaussianWhiteNoise(unittest.TestCase):
    
    def test_default_params(self):
        snr = 10.0
        noise_mean = 30
        obj = GaussianWhiteNoise(snr, noise_mean)

        self.assertEqual(snr, obj.SNR_db)
        self.assertEqual(noise_mean, obj.noise_mean)

    def test_generate_noise_signal_correct_var(self):
        obj = GaussianWhiteNoise(10, 0)

        variance = 10
        noise = obj.generate_noise_signal(variance, 100000)
        power = obj.get_required_noise_power(noise)

        self.assertAlmostEqual(1, power, 1)

    def test_singal_power_ones(self):
        obj = GaussianWhiteNoise(10, 0)
        
        signal = np.ones(10000)
        mean_power = obj.get_required_noise_power(signal)

        self.assertAlmostEqual(0.1, mean_power, 7)

    def test_signal_power_zeros(self):
        snr = 10
        linear_snr = 10.0**(snr/10.0)
        obj = GaussianWhiteNoise(snr, 0)

        signal = np.zeros(10000)
        power = obj.get_required_noise_power(signal)

        self.assertAlmostEqual(linear_snr, power, 7)

    def test_add_interference(self):
        SNR = 10
        obj = GaussianWhiteNoise(SNR, 0)

        signal = np.ones(10000000) * 2
        signal_with_noise = obj.add_interference(np.copy(signal))
        noise = signal_with_noise - signal

        signal_pow = np.mean((signal ** 2))
        noise_pow = np.mean((noise ** 2))

        new_snr = signal_pow/noise_pow

        self.assertAlmostEqual(SNR, new_snr, 1)