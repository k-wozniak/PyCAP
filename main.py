from PyCAP.recordingProbes.simple_recording_probe import SimpleRecordingProbe
from PyCAP.excitationSources.accurate_excitation_source import AccurateExcitationSource
from PyCAP.interferenceSources.gaussian_white_noise import GaussianWhiteNoise
from PyCAP.compoundElectrodes.overlapping_multipolar_electrodes import OverlappingMultipolarElectrodes
from PyCAP.excitationSources.simple_excitation_source import SimpleExcitationSource
from PyCAP.model.model_params import ModelParams
from PyCAP.model.model import Model
from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_qs

import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

import time
from scipy.io import savemat

# Generate cv distribution with unity
cv_min = 20
cv_max = 40
cv_step = 0.25

cv_dis = np.arange(cv_min, cv_max + cv_step, cv_step)
cv_dis = np.c_[cv_dis, np.ones(cv_dis.shape[0])]

#cv_dis = np.c_[cv_dis, cv_dis.copy()]

# Set model parameters
params = ModelParams(cv_dis, 0.030)
params.fs = 100000 # Hz

# Create model and add probes
model = Model(params)

# Generate Probes
excitation_source = SimpleExcitationSource(params.time_series)
excitation_source = AccurateExcitationSource(params.time_series)
model.add_excitation_source(excitation_source)

# Add noise
#white_noise = GaussianWhiteNoise(20.0)
#model.add_interference_source(white_noise)

# Add nine recording probes similar to the experimental data
probes = []
for i in range(9):
    recording_probe = SimpleRecordingProbe(0.08 + i*0.04)
    model.add_recording_probe(recording_probe)
    probes.append(recording_probe)

# Simulate noise
model.simulate()

# Generate Bipolar signals
bipolar = OverlappingMultipolarElectrodes([-1, 1])
bipolar.add_recording_probes(probes)
signals = bipolar.get_all_recordings()

file_name = "v2_flat_diss_20_40.mat"
savemat(file_name, {"signals": signals})

# Expected range of cv diss

#bipolar_qs = []
#for i in range(len(qs)-1):
#    bipolar_qs.append( qs[i+1] - qs[i] )

ws = []
""" bottom
for x in range(18, 30):
    cv_dis_w = np.arange(x, 60, 1)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)
    w = np.pad(w, (x-18, 0), 'constant')

    ws.append(w)

with xlsxwriter.Workbook('bottom_from_18_to_30_range_20-40.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(ws):
        worksheet.write_row(row_num, 0, data)
"""
""" top
for x in range(0, 11):
    cv_dis_w = np.arange(10, 40-x, 1)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)
    w = np.pad(w, (0, x), 'constant')

    ws.append(w)

with xlsxwriter.Workbook('top_from_40_to_30_range_20-40.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(ws):
        worksheet.write_row(row_num, 0, data)
"""

""" middle 

for x in range(1, 6):
    cv_dis_w = np.arange(10, 50, x)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)

    m = w.shape[0]
    w_zeros = np.zeros((1, x*m))
    w_zeros[:,::x] = w
    w_zeros = w_zeros.flatten()
    ws.append(w_zeros)

with xlsxwriter.Workbook('middle_from_1_to_6_range_20-40.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(ws):
        worksheet.write_row(row_num, 0, data)
"""


""" increased resolution """
"""
for x in [1, 0.5, 0.25]:
    cv_dis_w = np.arange(10, 50, x)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)

    ws.append(w)

with xlsxwriter.Workbook('increased_resolution_from_1_to_025_range_20-40.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(ws):
        worksheet.write_row(row_num, 0, data)
"""
'''
# Invert w bum christmas
for i in range(len(w)):
    v = cv_dis_w[i]
    w[i] = w[i] / (v**2)
'''

""" Find time 
times = []
for x in [750, 2000, 3000, 4000, 7500]:
    start = time.time()

    cv_dis_w = np.arange(10, x, 1)
    qs = generate_qs(probes, cv_dis_w, params.fs)
    w = be.NCap(signals, qs)
    
    time_taken = time.time() - start
    
    times.append([x, time_taken])

    print("next")

with xlsxwriter.Workbook('time_from_50_to_10000_run2.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(times):
        worksheet.write_row(row_num, 0, data)
"""
"""
#plt.plot(w)
#plt.show()

@profile(precision = 4)
def test_NCap(signas, probes, fs):
    cv_dis_w = np.arange(10, 200, 1)
    qs = generate_qs(probes, cv_dis_w, fs)
    w = be.NCap(signals, qs)
    return w

w = test_NCap(signals, probes, params.fs)

"""
# Display output
if recording_probe.is_output_set():
    for p in probes:
        plt.plot(p.output_signal)

    #plt.plot(probes[0].output_signal)
    #plt.plot(probes[8].output_signal)
    #plt.plot(signals[0])
    #plt.plot(probes[1].output_signal-probes[0].output_signal)
    plt.show()
