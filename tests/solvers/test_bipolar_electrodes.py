import unittest
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt

from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes
from PyCAP.solvers.bipolar_electrodes import two_cap, mean_two_cap, NCap

from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

class TestBipolarElectrodes(unittest.TestCase):

    first_probe: float = 0.08
    electrode_distance: float = 0.02
    fs: int = 100e3

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
        file = loadmat('interpStimCaps.mat', matlab_compatible=True)
        caps = file['interpEvent']

        # Set experimental parameters
        fs = 100e3
        du = 3.5e-3
        vmin = 5
        vmax = 100
        vstep = 0.5
        v_range = np.arange(vmin, vmax+vstep, vstep)
        interp_factor = 5

        # Create variables for VSR outputs
        num = int(((vmax+vstep) - vmin) * (1 / vstep))
        largest = np.zeros((num, caps.shape[2]))
        smallest = np.zeros((num, caps.shape[2]))

        for repeat in range(0, 10):
            (_, largest[:, repeat], smallest[:, repeat]) = VSR(caps[:, :, repeat], fs * interp_factor, du, vmin, vstep, vmax)

        plt.plot(v_range, np.mean(largest, axis=1))
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Amplitude (mV)')
        plt.show()
