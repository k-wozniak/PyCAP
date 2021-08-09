import unittest
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt

from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes, generate_q
from PyCAP.solvers.bipolar_electrodes import two_cap, mean_two_cap, NCap

from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model


class TestBipolarElectrodes(unittest.TestCase):

    # Simple distribution parameters
    first_probe: float = 0.08
    electrode_distance: float = 0.02
    fs: int = 100e3

    # Pig vagus (pv) data (Metcalfe et al 2018)
    file = loadmat('interpStimCaps.mat', matlab_compatible=True)
    caps_pv: np.ndarray = file['interpEvent']

    # Experimental parameters
    fs_pv: int = 100e3
    du_pv: float = 3.5e-3
    vmin_pv: int = 5
    vmax_pv: int = 100
    vstep_pv: float = 0.5
    v_range_pv: np.ndarray = np.arange(vmin_pv, vmax_pv + vstep_pv, vstep_pv)
    interp_factor_pv: int = 5

    def generate_test_signal_for_input_distribution(self, cv, recordings):
        # Random semi valid initial params
        params = ModelParams(cv, 0.030, self.fs)

        # simple perfect excitation source
        ses = SimpleExcitationSource(params.time_series)

        # Create required number of probes
        probes = []
        for i in range(0, recordings):
            probes.append(SimpleRecordingProbe(self.first_probe + i*self.electrode_distance))
        
        # Create model
        model = Model(params)
        model.add_excitation_source(ses)
        for p in probes:
            model.add_recording_probe(p)

        model.simulate()

        # Generate bipolar signals
        signals = []
        for i in range(0, len(probes)-1):
            signals.append(probes[i].output_signal - probes[i+1].output_signal)

        return signals, probes

    def test_two_cap_simple_distribution_correct_prediction(self):
        cv_dis = np.arange(50, 60, 1)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        signals, probes = self.generate_test_signal_for_input_distribution(cv_dis, 3)
        qs = generate_qs_from_probes(probes, cv_dis[:, 0], self.fs)

        w = two_cap(signals[0], signals[1], qs[0], qs[1], qs[1], qs[2])

        np.testing.assert_array_almost_equal(np.ones(w.shape[0])*0.1, w)

    def test_two_cap_pig_vagus_data(self):
        # Start with single repeat to begin with
        signals = self.caps_pv[:, :, 0]
        qs = generate_q(signals, self.v_range_pv)

    def test_mean_two_cap_simple_distribution_correct_prediction(self):
        cv_dis = np.arange(50, 60, 1)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        signals, probes = self.generate_test_signal_for_input_distribution(cv_dis, 5)
        qs = generate_qs_from_probes(probes, cv_dis[:, 0], self.fs)

        w = mean_two_cap(signals, qs)

        np.testing.assert_array_almost_equal(np.ones(w.shape[0])*0.1, w)
    
    def test_ncap_simple_distribution_correct_prediction(self):
        cv_dis = np.arange(50, 60, 1)
        cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

        signals, probes = self.generate_test_signal_for_input_distribution(cv_dis, 5)
        qs = generate_qs_from_probes(probes, cv_dis[:, 0], self.fs)

        w = NCap(signals, qs)

        np.testing.assert_array_almost_equal(np.ones(w.shape[0])*0.1, w)

    def test_ncap_uneven_distribution_correct_prediction(self):
        cv_range = np.arange(50, 60, 1)
        cv_weights = np.array([10, 20, 20, 5, 5, 0, 10, 20, 10, 0])
        cv_dis = np.c_[cv_range, cv_weights]

        signals, probes = self.generate_test_signal_for_input_distribution(cv_dis, 5)
        qs = generate_qs_from_probes(probes, cv_dis[:, 0], self.fs)

        w = NCap(signals, qs)

        expected_out = np.array([10, 20, 20, 5, 5, 0, 10, 20, 10, 0]) * 1e-2
        np.testing.assert_array_almost_equal(expected_out, w)

    def test_vsr_meanCAP_distribution(self):
        # Just plotting for now, to check what distribution looks like
        # TODO: Can change this to a proper test maybe by loading correct distribution from MATLAB (.mat file)?

        # Create variables for VSR outputs
        num = int(((self.vmax_pv+self.vstep_pv) - self.vmin_pv) * (1 / self.vstep_pv))
        largest = np.zeros((num, self.caps_pv.shape[2]))
        smallest = np.zeros((num, self.caps_pv.shape[2]))

        for repeat in range(0, 10):
            (_, largest[:, repeat], smallest[:, repeat]) = VSR(self.caps_pv[:, :, repeat], self.fs_pv
                                                               * self.interp_factor_pv, self.du_pv, self.vmin_pv,
                                                               self.vstep_pv, self.vmax_pv)

        plt.plot(self.v_range_pv, np.mean(largest, axis=1))
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Amplitude (mV)')
        plt.show()
