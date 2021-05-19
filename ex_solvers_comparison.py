from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
from PyCAP.solvers.utils import quadratic_solvers as qsolvers
from PyCAP.solvers.utils.qs_generation import generate_qs

import matplotlib.pyplot as plt
import numpy as np

import time

# Generate cv distribution with unity
cv_min = 20
cv_max = 40
cv_step = 1

steps = np.arange(cv_min, cv_max + cv_step, cv_step)
#cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

# For uniform distribution
cv_dis1 = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis1 = np.c_[cv_dis1, np.ones(cv_dis1.shape[0])]

# For step function
cv_dis2 = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis2 = np.c_[cv_dis2, steps.copy()]

# For a random function
cv_dis3 = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis3 = np.c_[cv_dis3, np.array([1, 1, 1, 1, 4, 4, 5, 4, 10, 10, 1, 2, 2, 1, 5, 6, 4, 4, 2, 2, 3])]


for cvdis in [cv_dis1, cv_dis2, cv_dis3]:
    # Set model parameters
    params = ModelParams(cvdis, 0.030)
    params.fs = 100000 # Hz

    # Create model and add probes
    model = Model(params)

    # Generate Probes
    excitation_source = SimpleExcitationSource(params.time_series)
    model.add_excitation_source(excitation_source)

    # Add nine recording probes similar to the experimental data
    probes = []
    for i in range(5):
        recording_probe = SimpleRecordingProbe(0.08 + i*0.04)
        model.add_recording_probe(recording_probe)
        probes.append(recording_probe)

    # Simulate noise
    model.simulate()

    # Generate Bipolar signals
    bipolar = OverlappingMultipolarElectrodes([-1, 1])
    bipolar.add_recording_probes(probes)
    signals = bipolar.get_all_recordings()

    cv_dis_w = np.arange(10, 100, 1)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    
    start = time.time()
    w_quad = be.NCap(signals, qs, None, qsolvers.quadratic_solver)
    quad_time = time.time() - start 

    start = time.time()
    w_cumm = be.NCap(signals, qs, None, qsolvers.cumminsolver_helper)
    cumm_time = time.time() - start

    #expected_output = np.pad(cvdis[:, 1], (10, 59))

    #difference_array = np.subtract(expected_output, w_cumm)
    #squared_array = np.square(difference_array)
    #mse1 = squared_array.mean()

    #difference_array = np.subtract(expected_output, w_quad)
    #squared_array = np.square(difference_array)
    #mse2 = squared_array.mean() 

    #print(w_cumm)
    #print(mse1)
    print(cumm_time)

    #print(w_quad)
    #print(mse2)
    print(quad_time)


