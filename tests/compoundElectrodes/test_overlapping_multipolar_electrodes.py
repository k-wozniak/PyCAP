import unittest
import numpy as np

from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.recordingProbes.recording_probe import RecordingProbe

class MockRP(RecordingProbe):
    def set_output_signal(self, signal: np.ndarray) -> None:
        self.output_signal = signal

class TestOverlappingMultipolarElectrodes(unittest.TestCase):

    def test_if_all_fields_correctly_initialised(self):
        bipolar_setup = [-1, 1]
        obj = OverlappingMultipolarElectrodes(bipolar_setup)

        self.assertEqual(bipolar_setup, obj.setup)
        self.assertIsNotNone(obj.recording_probes)

    def test_if_throw_error_for_negative_setups(self):
        self.assertRaises(ValueError, OverlappingMultipolarElectrodes.__init__, None, [])

    def test_add_recording_probe(self):
        obj = OverlappingMultipolarElectrodes([-1, 1])

        rp = MockRP()
        obj.add_recording_probe(rp)

        self.assertEqual(1, len(obj.recording_probes))

    def test_add_recording_probes(self):
        obj = OverlappingMultipolarElectrodes([-1, 1])
        rps = [MockRP(), MockRP(), MockRP()]

        obj.add_recording_probes(rps)

        self.assertEqual(3, len(obj.recording_probes))

    # Check the error for too long positions vs number of electrodes
    def test_get_recording_throws_error_if_negative_position(self):
        obj = OverlappingMultipolarElectrodes([-1, 1])
        electrodes = [MockRP(), MockRP(), MockRP()]
        
        obj.add_recording_probes(electrodes)

        self.assertRaises(ValueError, obj.get_recording, -1)

    def test_get_recording_throws_error_if_position_out_of_band(self):
        obj = OverlappingMultipolarElectrodes([-1, 1])
        electrodes = [MockRP(), MockRP(), MockRP()]

        obj.add_recording_probes(electrodes)

        self.assertRaises(ValueError, obj.get_recording, 3)

    def test_get_recording_throws_error_if_value_not_set(self):
        obj = OverlappingMultipolarElectrodes([-1, 1])
        electrodes = [MockRP(), MockRP(), MockRP()]

        obj.add_recording_probes(electrodes)

        self.assertRaises(ValueError, obj.get_recording, 0)

    def test_get_recording_simple_signals_bipolar(self):
        length = 5
        const = 5

        signals = [np.ones(length), np.zeros(length), np.ones(length)*const]
        electrodes = [MockRP(), MockRP(), MockRP()]

        for i in range(0, len(electrodes)):
            electrodes[i].set_output_signal(signals[i])

        obj = OverlappingMultipolarElectrodes([-1, 1])
        obj.add_recording_probes(electrodes)
        
        output = obj.get_all_recordings()
        
        self.assertEqual(2, np.shape(output)[0])
        np.testing.assert_array_equal(-1* np.ones(length), output[0])
        np.testing.assert_array_equal(np.ones(length) * const, output[1])

