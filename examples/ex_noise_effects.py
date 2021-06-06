from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.accurate_excitation_source import AccurateExcitationSource
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
from PyCAP.interferenceSources.brownian_noise import BrownianNoise
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
from scipy.io import savemat

# Generate cv distribution with unity
cv_min = 20
cv_max = 40
cv_step = 1

cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

# Set model parameters
params = ModelParams(cv_dis, 0.050)
params.fs = 100000 # Hz

s = params.number_of_cv_classes()

# Create model and add probes
model = Model(params)

# Generate Probes
excitation_source = SimpleExcitationSource(params.time_series)
model.add_excitation_source(excitation_source)

# Add nine recording probes similar to the experimental data
probes = []
for i in range(4):
    recording_probe = SimpleRecordingProbe(0.08 + i*0.04)
    model.add_recording_probe(recording_probe)
    probes.append(recording_probe)

ws = []

dbs = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
g_ranges = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

for x in dbs:
    # Add noise
    white_noise = GaussianWhiteNoise(x)

    #brownian_noise = BrownianNoise((-10, 10), (-x, x)) 

    model.add_interference_source(white_noise)

    # Simulate noise
    model.simulate()

    # Generate Bipolar signals
    bipolar = OverlappingMultipolarElectrodes([-1, 1])
    bipolar.add_recording_probes(probes)
    signals = bipolar.get_all_recordings()

    #plt.plot(signals[0])
    #plt.show()

    cv_dis_w = np.arange(10, 51, 1)
    qs = generate_qs_from_probes(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)
    w = np.append(w, x)

    ws.append(w)

    model.interference_sources = []
'''
brownian = GaussianWhiteNoise(10)
noise = brownian.generate_noise_signal(10, 1000)
ws.append(noise)
'''

savemat("simple_gaussian_noise_4_probes.xlsx",
    {
        "cv_diss": cv_dis,
        "fs": params.fs,
        "ws": ws,
    })