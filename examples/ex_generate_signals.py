from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.accurate_excitation_source import AccurateExcitationSource
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import PyCAP.solvers.utils.sfap_reconstruction as sfap_rec

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from scipy.stats import norm

file_name = "simulation_step_A(v)_dddd.mat"

# Generate cv distribution with unity
#cv_min, cv_max, cv_step = (30, 40, 1)

#cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
#cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])] # A uniform distribution
# cv_dis = np.c_[cv_dis, cv_dis.copy()] # A step distribution
# cv_dis = np.c_[cv_dis, np.array([1, 1, 1, 1, 4, 4, 5, 4, 10, 10, 1, 2, 2, 1, 5, 6, 4, 4, 2, 2, 3])] # A random distribution


cv_dis_int = np.arange(10 + 5, 74 + 1 + 5, 1)
cv_dis = np.arange(10 + 5, 74 + 0.1 + 5, 0.01)

cv_dis_vals = np.array([1765, 1218, 671, 247, 246, 244, 410, 1741, 5019, 11000, 20126, 31179, 42446, 51848, 58376, 60924, 60805, 57797, 53761, 47958, 42885, 37325, 33281, 29049, 27331, 24932, 24232, 22775, 22940, 21248, 21577, 19936, 19504, 17983, 17138, 15754, 13903, 12666, 10917, 9240, 7571, 6598, 4866, 3723, 3146, 1719, 1345, 658, 369, 251, 488, 488, 1011, 1421, 1628, 1335, 1809, 1629, 1329, 927, 927, 453, 743, 1267, 1901])
cv_dis_vals[-5:] = 0
cv_dis_vals[:5] = 0

cv_dis_vals = np.interp(cv_dis, cv_dis_int, cv_dis_vals)

#plt.bar(cv_dis, cv_dis_vals)
#plt.show()

cv_min, cv_max, cv_step = (10, 80, 1)
cv_dis_range = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = 100000*((2*norm.pdf(cv_dis_range,25,8)) + norm.pdf(cv_dis_range,50,6))
cv_dis_vals = np.round(cv_dis_vals)

plt.figure()
plt.bar(cv_dis_range, cv_dis_vals)
plt.show(block=False)

cv_dis = np.c_[cv_dis_range, cv_dis_vals]

# Set model parameters
#cv_dis = np.c_[np.array(80), np.array(1)]

params = ModelParams(cv_dis, 0.015)
params.fs = 100e3 # Hz

# Create model and add probes
model = Model(params)

# Generate Probes
excitation_source = SimpleExcitationSource(params.time_series)
#excitation_source = AccurateExcitationSource(params.time_series) # For A(v) = v^2
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

singular_signals = [s.output_signal for s in probes]

qs = generate_qs_from_probes(probes, np.arange(cv_min, cv_max + cv_step, cv_step), params.fs)  

#w = cv_dis_vals
w = cv_dis_vals #* (cv_dis_range**2)

As = []
for i in range(len(singular_signals)):
    As.append(sfap_rec.find_sfap_A(singular_signals[i], qs[i], w))

A1 = np.mean(As, axis=0)

As = []
for i in range(len(bipolar_signals)):
    As.append(sfap_rec.find_sfap_A(bipolar_signals[i], qs[i+1]-qs[i], w))

A2 = np.mean(As, axis=0)

sfap_shape = excitation_source.get_sfap(1)

plt.figure()
plt.plot(A2[:, 0])
plt.show(block=False)

plt.figure()
plt.plot(sfap_shape)
plt.show(block=False)

# Recreate A matrix
plt.figure()
#plt.plot(A1[:, 0])
i = 3
plt.plot((A2@(qs[i+1]-qs[i])@w))
plt.plot(bipolar_signals[i])
plt.show(block=False)

'''
plt.figure()
plt.plot(-singular_signals[0])
plt.plot(singular_signals[1])
plt.show(block=False)


fig, ax = plt.subplots(9)
for i in range(9):
    ax[i].plot(singular_signals[i])
plt.show(block=False)

#plt.plot(singular_signals[:, 0])
#plt.show()

plt.figure()
plt.plot(bipolar_signals.T[:700, 8])
plt.plot(bipolar_signals.T[:700, 0])
plt.show(block=False)
'''


"""
s = bipolar_signals[0].T
wss = 10
b = np.convolve(s, np.ones(wss), 'valid') / wss
plt.plot(b)
#plt.plot(singular_signals[0].T)
#plt.plot(singular_signals[1].T)
plt.show()

# Solve just in case
search_range = np.arange(10, 500, 1)

qs = generate_qs_from_probes(probes, search_range, params.fs)           
w = be.NCap(bipolar_signals, qs)

w_quad = w.copy()
for i in range(len(w)):
    v = search_range[i]
    w_quad[i] = w[i] / (v**2)

plt.bar(search_range, w)
#plt.plot(w)
plt.show()

savemat(file_name,
    {
        "cv_diss": cv_dis,
        "fs": params.fs,
        "probes_start": probes_start,
        "probes_center_to_center_distance": probes_center_to_center_distance,
        "number_of_probes": number_of_probes,
        
        "singular_signals": singular_signals,
        "bipolar_signals": bipolar_signals,
    })
"""
plt.show()