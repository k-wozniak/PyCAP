from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.accurate_excitation_source import AccurateExcitationSource
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

file_name = "simulation_step_A(v)_dddd.mat"

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (30, 40, 1)

cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])] # A uniform distribution
# cv_dis = np.c_[cv_dis, cv_dis.copy()] # A step distribution
# cv_dis = np.c_[cv_dis, np.array([1, 1, 1, 1, 4, 4, 5, 4, 10, 10, 1, 2, 2, 1, 5, 6, 4, 4, 2, 2, 3])] # A random distribution

#cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
#cv_dis_vals = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 315, 306, 273, 245, 215, 223, 169, 134, 105, 104, 83, 92, 67, 73, 68, 54, 59, 52, 60, 52, 36, 29, 35, 19, 19, 22, 16, 14, 18, 13, 7, 16, 16, 17, 8, 11, 15, 15, 8, 11, 15, 8, 9, 7, 13, 4, 7, 3, 7, 6, 7, 2, 2, 5, 2, 2, 4, 2, 4, 2, 2, 2, 1, 4, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 3, 1, 2, 3, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2]
#cv_dis_vals = np.array(cv_dis_vals) * 10

#cv_dis = np.c_[cv_dis, cv_dis_vals]

# Set model parameters
cv_dis = np.c_[np.array(30), np.array(1)]

params = ModelParams(cv_dis, 0.02)
params.fs = 100e3 # Hz

# Create model and add probes
model = Model(params)

# Generate Probes
excitation_source = SimpleExcitationSource(params.time_series)
# excitation_source = AccurateExcitationSource(params.time_series) # For A(v) = v^2
model.add_excitation_source(excitation_source)

# Add noise
#white_noise = GaussianWhiteNoise(20.0)
#model.add_interference_source(white_noise)

# Add nine recording probes similar to the experimental data
probes_start = 80e-3
probes_center_to_center_distance = 3.5e-3
number_of_probes = 10

probes_distances = []
probes = []
for i in range(number_of_probes):
    # Find the distance
    current_distance = probes_start + (i*probes_center_to_center_distance)
    
    # Generate the probe
    recording_probe = SimpleRecordingProbe(current_distance)
    
    # Add probe to the model
    model.add_recording_probe(recording_probe)
    
    # Save info
    probes.append(recording_probe)
    probes_distances.append(current_distance)

# Simulate noise
model.simulate()

# Generate Bipolar signals
bipolar = OverlappingMultipolarElectrodes([-1, 1])
bipolar.add_recording_probes(probes)
bipolar_signals = bipolar.get_all_recordings()

singular = OverlappingMultipolarElectrodes([1])
singular.add_recording_probes(probes)
singular_signals = singular.get_all_recordings()

s = bipolar_signals[0].T
wss = 10
b = np.convolve(s, np.ones(wss), 'valid') / wss
plt.plot(b)
#plt.plot(singular_signals[0].T)
#plt.plot(singular_signals[1].T)
plt.show()

# Solve just in case
search_range = np.arange(10, 120, 1)

qs = generate_qs_from_probes(probes, search_range, params.fs)           

w = be.NCap(bipolar_signals, qs)

w_quad = w.copy()
for i in range(len(w)):
    v = search_range[i]
    w_quad[i] = w[i] / (v**2)

w_VSR = be.VSR(singular_signals.T, params.fs, probes_center_to_center_distance, 10, 1, 121)

#plt.plot(w)
plt.plot(w_VSR)
plt.show()