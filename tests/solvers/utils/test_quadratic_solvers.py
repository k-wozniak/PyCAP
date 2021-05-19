from PyCAP.solvers.utils.quadratic_solvers import quadratic_solver, cumminsolver

import unittest
import numpy as np

class TestQuadraticSolver(unittest.TestCase):
    
    def test_quadratic_solver_simple_input_correct_output(self):     
        C = np.array([[2, 1], [1, 2]])

        w = quadratic_solver(C)

        np.testing.assert_array_almost_equal(w, np.array([0.5, 0.5]), 2)

    def test_quadratic_solver_simple_input_correct_output_2(self):     
        C = np.array([[-100, 50, 10], [100, -50, 10], [100, 50, -10]])

        w = quadratic_solver(C)

        np.testing.assert_array_almost_equal(w, np.array([0.05, 0.11, 0.84]), 2)

    def test_cumminsolver_singular_input(self):
        C = np.array([[2, 1], [1, 2]])
        
        w = cumminsolver(C)

        np.testing.assert_array_almost_equal(w, np.array([0.5, 0.5]), 2)

    def test_cumminsolver_singular_input_2(self):
        C = np.array([[-100, 50, 10], [100, -50, 10], [100, 50, -10]])

        w = cumminsolver(C, 0.000001, 100)

        # Something is not optimal with the solver
        np.testing.assert_array_almost_equal(w, np.array([0.05, 0.11, 0.84]), 2)
        
    def test_cumminsolver_singular_input_2_with_warm_start(self):
        C = np.array([[-100, 50, 10], [100, -50, 10], [100, 50, -10]])

        warm_start = np.matrix([0.05, 0.11, 0.84]).T

        w = cumminsolver(C, 0.000001, 100, warm_start)

        np.testing.assert_array_almost_equal(w, np.array([0.05, 0.11, 0.84]), 2)



        