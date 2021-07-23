import PyCAP.solvers.utils.sfap_reconstruction as reconstruct
from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

import unittest
import numpy as np
import xlsxwriter

import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

class TestSFAPReconstruction(unittest.TestCase):

    def test_find_matrix_A_simple_input_correct_output(self):
        C = np.array([[1], [2], [3]])
        w = np.array([[0.1], [0.5], [0.4]])
        q = np.array([[1,1,1], [2,2,2], [3,3,3]])

        sfaps = reconstruct.find_matrix_A(C, q, w)

        np.testing.assert_almost_equal(C, np.matmul(sfaps, np.matmul(q, w)))

    def test_find_matrix_A_from_set_simple_identical_sets(self):
        C = np.array([[1], [2], [3]])
        w = np.array([[0.1], [0.5], [0.4]])
        q = np.array([[1,1,1], [2,2,2], [3,3,3]])

        Cs = []
        qs = []
        for _ in range(10):
            Cs.append(np.copy(C))
            qs.append(np.copy(q))

        sfaps = reconstruct.find_matrix_A_from_set(Cs, qs, w)

        np.testing.assert_almost_equal(C, np.matmul(sfaps, np.matmul(q, w)))

    def test_recreate_A_matrix_input_zeros_out_zeros(self):
        dim = 5
        z = np.zeros((dim, dim))
        z_ref = np.zeros((dim, dim))

        z = reconstruct.recreate_A_matrix(z)

        np.testing.assert_almost_equal(z, z_ref)

    def test_recreate_A_matrix_input_ones_out_ones(self):
        dim = 5
        z = np.ones((dim, dim))
        z_ref = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ])

        z = reconstruct.recreate_A_matrix(z)

        np.testing.assert_almost_equal(z, z_ref)

    def test_recreate_A_matrix_input_simple_out_as_predicted(self):
        z = np.array([
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [3, 2, 1, 0, 0],
            [4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1],
            [6, 5, 4, 3, 2],
            [7, 6, 5, 4, 3],
            [0, 7, 6, 5, 4],
            [0, 0, 7, 6, 5],
            [0, 0, 0, 7, 6],
            [0, 0, 0, 0, 7],
        ])
        z_ref = np.array([
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [3, 2, 1, 0, 0],
            [4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1],
            [6, 5, 4, 3, 2],
            [7, 6, 5, 4, 3],
            [0, 7, 6, 5, 4],
            [0, 0, 7, 6, 5],
            [0, 0, 0, 7, 6],
            [0, 0, 0, 0, 7],
        ])

        z = reconstruct.recreate_A_matrix(z)

        np.testing.assert_almost_equal(z, z_ref)

    def test_recreate_A_matrix_boundary_ones_everything_else_zeros(self):
        """ This test is weird as it shows kind of a weird 'bug'. As it is 
            unknown how long the signal is, especially in a square matrix, 
            it is assumed that the bottom part of the array is a signal and 
            is not replaced necessary with zeros. Generally it should not be
            an issues but the behaviour should be noted. """
        z = np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])
        z_ref = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])

        z = reconstruct.recreate_A_matrix(z)

        np.testing.assert_almost_equal(z, z_ref)

    def test_find_matrix_A_from_set_simple_matrix_correct_output(self):
        signal = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        q = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        w = [1, 1, 1]

        expected_A = reconstruct.find_matrix_A(signal, q, w)

        signals = [signal.copy(), signal.copy(), signal.copy(), signal.copy(), signal.copy()]
        qs = [q.copy(), q.copy(), q.copy(), q.copy(), q.copy()]
        
        A = reconstruct.find_matrix_A_from_set(signals, qs, w)

        np.testing.assert_almost_equal(expected_A, A)
    
    def test_find_matrix_A_from_set_generated_signal_correct_sfaps(self):
        # Generate cv distribution with unity
        cv_min = 30
        cv_max = 40
        cv_step = 1

        cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        # Set model parameters
        params = ModelParams(cv_dis, 0.010)
        params.fs = 100000 # Hz

        model = Model(params)
        model.add_excitation_source(SimpleExcitationSource(params.time_series))

        probes = []
        for i in range(9):
            recording_probe = SimpleRecordingProbe(0.02 + i*0.02)
            model.add_recording_probe(recording_probe)
            probes.append(recording_probe)
        
        # Generate Signals
        model.simulate()
        
        # Generate Bipolar signals
        bipolar = OverlappingMultipolarElectrodes([-1, 1])
        bipolar.add_recording_probes(probes)
        signals = bipolar.get_all_recordings()

        # Find w and Qs
        probabilities = (np.ones(cv_dis.shape[0]) * 1/cv_dis.shape[0]).T

        qs = generate_qs_from_probes(probes, cv_dis[:, 0], params.fs)
        bipolar_qs = []
        for i in range(len(qs)-1):
            bipolar_qs.append( qs[i+1] - qs[i] )

        signal = signals[0] + signals[2] + signals[4] + signals[6]
        total_q = bipolar_qs[0] + bipolar_qs[2] + bipolar_qs[4] + bipolar_qs[6]

        A_simple = reconstruct.find_matrix_A(signals[0], bipolar_qs[0], probabilities)
        
        A = reconstruct.find_matrix_A(signal, total_q, probabilities)
        A2 = reconstruct.find_matrix_A_from_set(signals, bipolar_qs, probabilities)

        #Asing = reconstruct.find_matrix_A(signals[0], bipolar_qs[0], probabilities)

        signal_0 = np.array(np.matmul(np.matmul(A_simple, bipolar_qs[0]), probabilities)).flatten()

        #sig1 = np.matmul(s1, probabilities)

        ses = SimpleExcitationSource(params.time_series)
        sfap = ses.get_sfap(1, 0)

        time_series = range(0, 1000)

        A = reconstruct.recreate_A_matrix(A)
        A2 = reconstruct.recreate_A_matrix(A2)

        AQ = np.matmul(A, np.asmatrix(bipolar_qs[0]))
        reconstructed_signal1 = np.matmul(AQ, probabilities)
        
        AQ2 = np.matmul(A2, np.asmatrix(bipolar_qs[0]))
        reconstructed_signal2 = np.matmul(AQ2, probabilities)

        fig, axs = plt.subplots(5)
        #axs[0].plot(np.matmul(A, bipolar_qs[0]))
        #axs[1].plot(np.matmul(A, bipolar_qs[1]))
        #axs[2].plot(np.matmul(A, bipolar_qs[2]))
        axs[0].plot(time_series, signals[0])
        axs[1].plot(time_series, signal_0)
        axs[2].plot(time_series, reconstructed_signal2.T)
        axs[3].plot(time_series, A[:, 0])
        axs[4].plot(A2)

        plt.show()

        with xlsxwriter.Workbook('amatrix.xlsx') as workbook:
            worksheet = workbook.add_worksheet()

            for row_num, data in enumerate([signals[0], signal_0]):
                worksheet.write_row(row_num, 0, data)
    
    def test_the_principle(self):
        C = np.array([[1], [2], [3]])
        w = np.array([[0.1], [0.5], [0.4]])
        q = np.array([[1,1,1], [2,2,2], [3,3,3]])

        sfaps = reconstruct.find_sfap_A(C, q, w)

        np.allclose(np.dot(q@w, sfaps), C)