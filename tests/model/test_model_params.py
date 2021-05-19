import unittest
import numpy as np

from PyCAP.model.model_params import ModelParams

class TestModelParams(unittest.TestCase):
    
    def simple_dis(self):
        cv_dis = np.arange(10, 120, 1)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        return np.copy(cv_dis)

    def test_default_values_set(self):
        simulation_length = 0.03
        obj = ModelParams(self.simple_dis(), simulation_length)

        self.assertIsNotNone(obj.fs)
        self.assertIsNotNone(obj.cv_distribution)

        self.assertEqual(simulation_length, obj.simulation_length)
        self.assertIsNotNone(obj.time_series)

    def test_constructor(self):
        simulation_length = 0.05
        fs = 10
        time_series = np.arange(0, simulation_length, 1/fs)

        obj = ModelParams(self.simple_dis(), simulation_length, fs)

        self.assertEqual(simulation_length, obj.simulation_length)
        self.assertEqual(fs, obj.fs)
        self.assertEqual(len(time_series), len(obj.time_series))

    def test_returned_number_of_cv_classes(self):
        simulation_length = 0.05
        obj = ModelParams(self.simple_dis(), simulation_length)

        self.assertEqual(110, obj.number_of_cv_classes())

    def test_cv_distribution(self):
        obj = ModelParams(self.simple_dis(), 0.05)

        cv_diss = obj.get_cv_distribution()

        self.assertEqual(110, len(cv_diss))

        for i in range(0, cv_diss.shape[0]):
            v = cv_diss[i][0]
            c = cv_diss[i][1]

            self.assertEqual(i+10, v)
            self.assertEqual(1, c)

