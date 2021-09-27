from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
from PyCAP.interferenceSources.brownian_noise import BrownianNoise
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs_from_probes

import numpy as np
import matplotlib.pyplot as plt

# Generate cv distribution with unity
cv_min, cv_max, cv_step = (20, 40, 1)
cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

# Set model parameters
params = ModelParams(cv_dis, 0.050)
params.fs = 100e3 # Hz

# Create model and add probes
model = Model(params)

excitation_source = SimpleExcitationSource(params.time_series)
model.add_excitation_source(excitation_source)

# Add nine recording probes
probes = []
for i in range(9):
    recording_probe = SimpleRecordingProbe(0.08 + i*0.04)
    model.add_recording_probe(recording_probe)
    probes.append(recording_probe)

ws = []
dbs = [-5, 5, 15, 25]

for db in dbs:
    # resent the model interface sources
    model.interference_sources = []

    # Add noise
    white_noise = GaussianWhiteNoise(db)
    model.add_interference_source(white_noise)
    model.simulate()

    # Generate Bipolar signals
    bipolar = OverlappingMultipolarElectrodes([-1, 1])
    bipolar.add_recording_probes(probes)
    signals = bipolar.get_all_recordings()

    # Solve
    search_range = np.arange(10, 51, 1)
    qs = generate_qs_from_probes(probes, search_range, params.fs)
    w = be.NCap(signals, qs)

    plt.figure("W found at db = " + str(db))
    plt.xlabel("Velocity CV m/s")
    plt.ylabel("Percentage contribution")
    plt.bar(search_range, w)
    plt.show(block=False)


plt.show()
