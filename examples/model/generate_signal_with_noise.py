from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
import PyCAP.excitationSources as es
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Natural distribution for fibres
cv_min, cv_max, cv_step = (10, 80, 1)
cv_dis_range = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = 100000*((2*norm.pdf(cv_dis_range,25,8)) + norm.pdf(cv_dis_range,50,6))
cv_dis_vals = np.round(cv_dis_vals) # To ensure they are all int

plt.figure("Nerve fibre distribution")
plt.xlabel('Conduction Velocity (CV) m/s')
plt.ylabel('Fibres Count')
plt.plot(cv_dis_range, cv_dis_vals);
plt.show(block=False)

# Set model parameters
cv_dis = np.c_[cv_dis_range, cv_dis_vals]
params = ModelParams(cv_dis, 0.015, 400e3)

# Create model and add probes
model = Model(params)

# Generate Probes
excitation_source = es.AccurateExcitationSource(params.time_series) # For A(v) = v^2
model.add_excitation_source(excitation_source)

# Add nine recording probes similar to the experimental data
probes_start = 80e-3
probes_center_to_center_distance = 3.5e-3
number_of_probes = 3

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

# Add noise
white_noise = GaussianWhiteNoise(20.0)
model.add_interference_source(white_noise)

# Simulate noise
model.simulate()

singular_signals = [s.output_signal for s in probes]

# Generate Bipolar signals
bipolar = OverlappingMultipolarElectrodes([-1, 1])
bipolar.add_recording_probes(probes)
bipolar_signals = bipolar.get_all_recordings()

# Plot singular signals obtained
fig, ax = plt.subplots(number_of_probes)
for i in range(number_of_probes):
    ax[i].set_title("Singular Signals Probe " + str(i))
    ax[i].set_xlabel("Time s")
    ax[i].set_ylabel("Amplitude")
    ax[i].plot(singular_signals[i])
plt.show(block=False)

# Plot bipolar signals obtained
fig, ax = plt.subplots(number_of_probes - 1) # -1 because it's bipolar
for i in range(number_of_probes - 1):
    ax[i].set_title("Bipolar Signals Probe " + str(i))
    ax[i].set_xlabel("Time s")
    ax[i].set_ylabel("Amplitude")
    ax[i].plot(bipolar_signals[i])
plt.show(block=False)

plt.show()