import PyCAP.solvers.utils.sfap_reconstruction as sfap_rec
import PyCAP.excitationSources as es
from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
import matplotlib.pyplot as plt

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (30, 40, 1)
cv_dis_range = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis_vals = np.ones(cv_dis_range.shape[0]) # A uniform distribution

# Set model parameters
cv_dis = np.c_[cv_dis_range, cv_dis_vals]
params = ModelParams(cv_dis, 0.010, 100e3)

# Create model and add probes
model = Model(params)

# Generate Probes
#excitation_source = es.SimpleExcitationSource(params.time_series)
excitation_source = es.AccurateExcitationSource(params.time_series) # For A(v) = v^2
#excitation_source = es.PyPnsLike(params.time_series)

model.add_excitation_source(excitation_source)

# Add 10 recording probes similar to the experimental data
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

singular_signals = [s.output_signal for s in probes]
qs = generate_qs_from_probes(probes, cv_dis_range, params.fs)  

#w = cv_dis_vals
# If signal is created using AccurateExcitationSource
# then the output w will be scaled up
w = cv_dis_vals * (cv_dis_range**2) 

A = sfap_rec.find_sfap_A_from_a_set(singular_signals, qs, w, 100)

sfap_shape = excitation_source.get_sfap(cv_min)

plt.figure("Reconstructed SFAP shape")
plt.plot(A[:, 0])
plt.show(block=False)

plt.figure("Original SFAP shape")
plt.plot(sfap_shape)
plt.show(block=False)

# Recreate A matrix
plt.figure("Singular original and reconstructed signal")
plt.plot((A@qs[0]@w))
plt.plot(singular_signals[0])
plt.show(block=False)

plt.show()