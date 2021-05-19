import PyCAP.solvers.utils.qs_generation as gen
from PyCAP.recordingProbes.recording_probe import RecordingProbe

import unittest
import numpy as np

class MockProbe(RecordingProbe):
    def __init__(self, pos):
        self.position = pos
        self.output_signal = np.arange(0, 200, 1)

    def set_output_signal(self, signal: np.ndarray) -> None:
        self.output_signal = signal

class TestQsGeneration(unittest.TestCase):
    
    def test_simple_input_correct_output(self):     
        const = 10
        probe = MockProbe(const)
        fs = const
        velocities = np.arange(1, 3, 1)

        q = gen.generate_q_matrix(probe, velocities, fs)

        self.assertEqual((200, 2), q.shape)
        self.assertEqual(1, q[100, 0])
        self.assertEqual(1, q[50, 1])
       
    def test_recording_probe_without_signal_throws_error(self):
        probe = MockProbe(10)
        probe.output_signal = None

        velocities = np.arange(1, 3, 1)

        self.assertRaises(ValueError, gen.generate_q_matrix, probe, velocities, 10)

    def test_simple_input_correct_output_for_multiple_probes(self):
        probes_num = 3
        const = 10
        fs = const
        velocities = np.arange(1, 3, 1)
        
        probes = []
        for _ in range(probes_num):
            probe = MockProbe(const)
            probes.append(probe)

        qs = gen.generate_qs(probes, velocities, fs)

        for i in range(probes_num):
            q = qs[i]

            self.assertEqual((200, 2), q.shape)
            self.assertEqual(1, q[100, 0])
            self.assertEqual(1, q[50, 1])