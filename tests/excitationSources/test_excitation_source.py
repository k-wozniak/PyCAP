import unittest
import numpy as np

from PyCAP.excitationSources.excitation_source import ExcitationSource

class MockExcitationSource(ExcitationSource):
    """ As the tested class is abstract the methods need to be overwritten
        before testing is possible. """
    def get_sfap(self, velocity: float, time_shift: int = 0) -> np.ndarray:
        return np.empty(0)


class TestExcitationSource(unittest.TestCase):

    def test_if_all_fields_correctly_initialised(self):
        obj = MockExcitationSource()

        self.assertIsNotNone(obj.start)
        self.assertIsNotNone(obj.is_continuous)
        self.assertIsNotNone(obj.position)

    def test_if_setting_position_works(self):
        obj = MockExcitationSource()

        pos = 0.085
        obj.position = pos

        self.assertEqual(pos, obj.get_position())