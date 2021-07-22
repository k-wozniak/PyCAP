import unittest
import numpy as np

from numpy.testing import assert_array_equal

from PyCAP.io.load import StimulationData

raw_data = "rawdata"
sample_rate = "sampleRate"
stimulus_pulse = "stimulusPulse"

example_data = {
        "sampleRate": np.array([[10]]),
        "rawdata": np.array([
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [3, 2, 1, 0],
            [4, 3, 2, 1],
            [5, 1, 3, 2],
            [6, 0, 1, 3],
            [7, 0, 0, 1],
            [8, 0, 0, 0],
            [9, 0, 0, 0],
        ]),
        "stimulusPulse": np.array([[0], [3], [0], [0], [0], [0], [0], [0], [0]]),
    }

class TestStimulationData(unittest.TestCase):    
    def test_correctly_identified_stimulations(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [3], [0], [3], [0], [3], [0], [3]])
        out = np.array([0, 2, 4, 6, 8])

        sd = StimulationData(d, remove_from_start=0)

        assert_array_equal(out, sd.stimulation_locations)

    def test_correctly_identified_clamped_stimulations(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [3], [3], [3], [0], [3], [3], [3]])
        out = np.array([0, 2, 6])

        sd = StimulationData(d, remove_from_start=0)

        assert_array_equal(out, sd.stimulation_locations)

    def test_correctly_identified_fluctuating_stimulations(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0.5], [2.9], [3], [2.2], [0.1], [3.1], [3], [2.9]])
        out = np.array([0, 2, 6])

        sd = StimulationData(d, remove_from_start=0)

        assert_array_equal(out, sd.stimulation_locations)

    def test_remove_correct_number_of_pulses_removed_from_the_beginning_1(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [3], [0], [3], [0], [3], [0], [3]])
        out = np.array([2, 4, 6, 8])

        sd = StimulationData(d, remove_from_start=1)

        assert_array_equal(out, sd.stimulation_locations)

    def test_remove_correct_number_of_pulses_removed_from_the_beginning_2(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [3], [0], [3], [0], [3], [0], [3]])
        out = np.array([4, 6, 8])

        sd = StimulationData(d, remove_from_start=2)

        assert_array_equal(out, sd.stimulation_locations)

    def test_remove_correct_number_of_pulses_removed_from_the_beginning_3(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [3], [0], [3], [0], [3], [0], [3]])
        out = np.array([6, 8])

        sd = StimulationData(d, remove_from_start=3)

        assert_array_equal(out, sd.stimulation_locations)

    def test_if_correct_sample_rate(self):
        out_fs = example_data[sample_rate].copy()

        sd = StimulationData(example_data, remove_from_start=0)

        self.assertEqual(out_fs, sd.sample_rate)

    def test_getting_signals(self):
        d = example_data.copy()
        d[stimulus_pulse] = np.array([[3], [0], [0], [0], [3], [0], [0], [0], [0]])
        
        sd = StimulationData(example_data.copy(), remove_from_start=0)
        
        out = sd.get_signals(0, 1, 3, normalise=False)

        expected_out = [np.array([
            [1, 2, 3],
            [0, 1, 2],
            [0, 0, 1],
        ])]

        assert_array_equal(expected_out, out)