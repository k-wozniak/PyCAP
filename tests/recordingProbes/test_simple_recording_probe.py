import unittest
import numpy as np

from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe

class TestSimpleRecordingProbe(unittest.TestCase):

    def test_constructors(self):
        pos = 888
        
        obj = SimpleRecordingProbe(pos)
        self.assertEqual(pos, obj.get_position())

    def test_check_if_setting_signal_works(self):
        obj = SimpleRecordingProbe(1)

        sig = np.empty(11)
        obj.output_signal = sig

        self.assertTrue(obj.is_output_set())
