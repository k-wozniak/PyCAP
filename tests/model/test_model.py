import unittest
import numpy as np
from unittest.mock import MagicMock

from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

class TestModel(unittest.TestCase):
    def default_params(self):
        cv_dis = np.arange(10, 120, 1)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        params = ModelParams(cv_dis, 0.05)
        
        return params

    def test_default_values_set(self):
        m = Model(self.default_params())
        
        self.assertIsNotNone(m.params)
        self.assertIsNotNone(m.excitation_sources)
        self.assertIsNotNone(m.recording_probes)
        self.assertIsNotNone(m.interference_sources)

    def test_adding_excitation_source(self):
        m = Model(self.default_params())
        obj = 1

        m.add_excitation_source(obj)

        self.assertEqual(1, len(m.excitation_sources))

    def test_adding_recording_probe(self):
        m = Model(self.default_params())
        obj = 1

        m.add_recording_probe(obj)

        self.assertEqual(1, len(m.recording_probes))
    
    def test_adding_interference_source(self):
        m = Model(self.default_params())
        obj = 1

        m.add_interference_source(obj)

        self.assertEqual(1, len(m.interference_sources))
