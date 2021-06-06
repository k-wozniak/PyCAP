import unittest

from PyCAP.solvers.recorded_signals import RecordedSignals

class TestRecordedSignals(unittest.TestCase):
    
    def test_generate_simple_objects_stores_data_correctly(self):
        fs = 100_000
        distances = [1, 2, 3]
        signals = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        r = RecordedSignals(fs, distances, signals)

        self.assertEqual(fs, r.fs)
        self.assertEqual(3, len(r.data))

    def test_throws_error_when_different_arrays_sizes(self):
        fs = 100_000
        distances = [1, 2, 3]
        signals = [[0, 1, 2], [3, 4, 5]]

        self.assertRaises(Exception, RecordedSignals.__init__, None, fs, distances, signals)

    def test_throws_error_when_different_signals_sizes(self):
        fs = 100_000
        distances = [1, 2, 3]
        signals = [[0, 1, 2], [3, 4, 5, 6], [6, 7, 8, 9]]

        self.assertRaises(Exception, RecordedSignals.__init__, None, fs, distances, signals)

    def test_correct_rounding(self):
        fs = 100_000
        distances = [0.000001, 0.0000019]
        signals = [[0, 1, 2], [3, 4, 5]]

        r = RecordedSignals(fs, distances, signals)

        self.assertEqual(signals[0], r.data[0.000001])
        self.assertEqual(signals[1], r.data[0.000002])

    def test_adding_signal(self):
        r = RecordedSignals(100_000, [], [])
        
        d = 0.05
        s = [1, 2, 3]

        r.add_signal(d, s)

        self.assertEqual(1, len(r.data))
        self.assertEqual([1, 2, 3], r.data[d])

    def test_remove_signal_removed(self):
        fs = 100_000
        distances = [1, 2, 3]
        signals = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        r = RecordedSignals(fs, distances, signals)
        removed = r.remove_signal(1)

        self.assertTrue(removed)
        self.assertEqual(2, len(r.data))

    def test_remove_signal_return_false_if_does_not_exist(self):
        fs = 100_000
        distances = [1, 2, 3]
        signals = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        r = RecordedSignals(fs, distances, signals)

        self.assertFalse(r.remove_signal(34))
