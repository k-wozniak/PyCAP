import unittest
import numpy as np

from PyCAP.recordingProbes.recording_probe import RecordingProbe

class MockRecordingProbe(RecordingProbe):
    """ As the tested class is abstract the methods need to be overwriteed
        before testing is possible. """
    def set_output_signal(self, signal: np.ndarray) -> None:
        self.output_signal = signal

class TestRecordingProbe(unittest.TestCase):

    def test_if_all_fields_correctly_initialised(self):
        obj = MockRecordingProbe()

        self.assertIsNotNone(obj.position)
        self.assertIsNone(obj.output_signal)

    def test_if_output_is_set_function_works(self):
        obj = MockRecordingProbe()
        obj.output_signal = np.empty(11)

        self.assertTrue(obj.is_output_set())
