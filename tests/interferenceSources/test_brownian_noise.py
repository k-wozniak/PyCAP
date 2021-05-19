import unittest
import numpy as np

from PyCAP.interferenceSources.brownian_noise import BrownianNoise

class TestBrownianNoise(unittest.TestCase):
    
    def test_default_params(self):
        generation_range = (-5, 5)
        final_range = (-1, 1)
        obj = BrownianNoise(generation_range, final_range)

        self.assertEqual(generation_range, obj.g_range)
        self.assertEqual(final_range, obj.f_range)

    def test_normalise_zeros_get_zeros(self):
        generation_range = (-5, 5)
        final_range = (-1, 1)
        obj = BrownianNoise(generation_range, final_range)

        arr = np.zeros(100)

        self.assertRaises(ValueError, obj.normalise, arr, -1, 1)

    def test_normalise_zeros_get_ones(self):
        generation_range = (-5, 5)
        final_range = (-1, 1)
        obj = BrownianNoise(generation_range, final_range)

        arr = np.arange(1, 100, 1)
        arr2 = np.arange(1, 100, 1)

        out = obj.normalise(arr, 1, 100)

        np.testing.assert_almost_equal(arr2, out, 0)

    def test_normalise_always_in_bounds(self):
        generation_range = (-5, 5)
        final_range = (-1, 1)
        obj = BrownianNoise(generation_range, final_range)

        arr = np.arange(-1000, 1000, 1)
        out = obj.normalise(arr, 5, 6)

        self.assertTrue(np.max(out) <= 6)
        self.assertTrue(np.min(out) >= 5)

    def test_generate_noise_signal_in_bounds(self):
        generation_range = (-5, 5)
        final_range = (-100, 100)
        obj = BrownianNoise(generation_range, final_range)

        out = obj.generate_noise_signal(10000)

        self.assertTrue(np.max(out) <= final_range[1])
        self.assertTrue(np.min(out) >= final_range[0])

