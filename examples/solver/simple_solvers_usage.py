from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model

from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
import matplotlib.pyplot as plt

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (30, 40, 1)
cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = np.ones(cv_dis.shape[0]) # A uniform distribution
cv_dis = np.c_[cv_dis, cv_dis_vals]

params = ModelParams(cv_dis, 0.02)
params.fs = 100e3 # Hz

model = Model(params) # Create model and add probes
excitation_source = SimpleExcitationSource(params.time_series)
model.add_excitation_source(excitation_source)

# Add 3 recording probes similar to the experimental data
probes_start = 80e-3
probes_center_to_center_distance = 3.5e-3
number_of_probes = 9

probes_distances = []
probes = []
for i in range(number_of_probes):
    current_distance = probes_start + (i*probes_center_to_center_distance)
    recording_probe = SimpleRecordingProbe(current_distance)
    model.add_recording_probe(recording_probe)

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

plt.figure("Bipolar signals generated")
plt.plot(bipolar_signals[0])
plt.plot(bipolar_signals[1])
plt.plot(bipolar_signals[2])
plt.show(block=False)

plt.figure("NCAP Distribution")
plt.bar(search_range, w)
plt.show()