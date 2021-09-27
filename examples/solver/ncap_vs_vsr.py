from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
import matplotlib.pyplot as plt

"""
The aim of this example is to show the difference between NCAP and VSR
On a set of generated data using the model avaliable thus controlling as
much of variables as possible.

The file is divided into following sections:
1. Signals generation using a simple excitation source
2. Finding the nerve distribution using:
    a. NCAP
    b. VSR
3. Plotting the results
"""

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (30, 40, 1)
cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = np.ones(cv_dis.shape[0]) # A uniform distribution

cv_dis = np.c_[cv_dis, cv_dis_vals]
#cv_dis = np.c_[np.array(30), np.array(1)] # A single fibre distribution

params = ModelParams(cv_dis, 0.02)
params.fs = 100e3 # Hz

model = Model(params) # Create model and add probes
excitation_source = SimpleExcitationSource(params.time_series)
model.add_excitation_source(excitation_source)

# Add 3 recording probes similar to the experimental data
probes_start = 80e-3
probes_center_to_center_distance = 3.5e-3
number_of_probes = 4

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


# Solve just in case
cv_min, cv_max, cv_step = (10, 80, 1)
search_range = np.arange(cv_min, cv_max, cv_step)

qs = generate_qs_from_probes(probes, search_range, params.fs)           
w = be.NCap(bipolar_signals, qs)
w_VSR = be.VSR(bipolar_signals, params.fs, probes_center_to_center_distance, cv_min, cv_step, cv_max-1)

plt.figure("Bipolar signals generated")
plt.plot(bipolar_signals[0])
plt.plot(bipolar_signals[1])
plt.plot(bipolar_signals[2])
plt.show(block=False)

plt.figure("NCAP Distribution")
plt.bar(search_range, w)
plt.show(block = False)

plt.figure("VSR Distribution")
plt.plot(search_range, w_VSR[0][0])
plt.plot(search_range, w_VSR[0][1])
plt.plot(search_range, w_VSR[0][2])
plt.show()