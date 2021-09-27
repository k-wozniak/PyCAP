from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
import PyCAP.excitationSources as es
import PyCAP.recordingProbes as rp
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes

import numpy as np
import matplotlib.pyplot as plt

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (25, 40, 1)
cv_dis_range = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = np.c_[cv_dis_range, np.ones(cv_dis_range.shape[0])] # A uniform distribution

# Set model parameters
cv_dis = np.c_[cv_dis_range, cv_dis_vals]
params = ModelParams(cv_dis, 0.005, 400e3)

# Create model and add probes
model = Model(params)

# Generate Excitation source (may be multipel sources)
excitation_source = es.SimpleExcitationSource(params.time_series)

model.add_excitation_source(excitation_source)

# Potentially add noise but not in this simulation
# white_noise = GaussianWhiteNoise(20.0)
# model.add_interference_source(white_noise)

# Add 3 recording probes
probes_start = 80e-3
probes_center_to_center_distance = 3.5e-3
number_of_probes = 3

probes_distances = []
probes = []
for i in range(number_of_probes):
    # Find the distance
    current_distance = probes_start + (i*probes_center_to_center_distance)
    
    # Generate the probe
    recording_probe = rp.SimpleRecordingProbe(current_distance)
    
    # Add probe to the model
    model.add_recording_probe(recording_probe)
    
    # Save info
    probes.append(recording_probe)
    probes_distances.append(current_distance)

# Simulate noise
model.simulate()

# Get singular signals
singular_signals = [s.output_signal for s in probes]

# Generate Bipolar signals (Optional)
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