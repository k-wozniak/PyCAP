import unittest
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt

from PyCAP.solvers.utils.qs_generation import generate_qs
from PyCAP.solvers.bipolar_electrodes import two_cap, mean_two_cap, NCap

# Pig vagus (pv) data (Metcalfe et al 2018)
file = loadmat('interpStimCaps.mat', matlab_compatible=True)
caps_pv = file['interpEvent']

# Experimental parameters
fs_pv = 100e3
du_pv = 3.5e-3
vmin_pv = 5
vmax_pv = 100
vstep_pv = 0.5
v_range_pv = np.arange(vmin_pv, vmax_pv + vstep_pv, vstep_pv)
interp_factor_pv = 5

len_recording = caps_pv.shape[0]
repeats = caps_pv.shape[2]

ch1 = 7
ch2 = 8
start_point = 100

c1_repeats = np.zeros((caps_pv.shape[0] - start_point, repeats))
c2_repeats = np.zeros((caps_pv.shape[0] - start_point, repeats))
for i in range(repeats):
    c1_repeats[:, i] = caps_pv[start_point:, ch1, i]
    c1_repeats[:, i] = c1_repeats[:, i] - np.mean(c1_repeats[:, i])

    c2_repeats[:, i] = caps_pv[start_point:, ch2, i]
    c2_repeats[:, i] = c2_repeats[:, i] - np.mean(c2_repeats[:, i])

# Distances
distances = [80e-3, 80e-3 + (ch1 * du_pv), 80e-3 + (ch2 * du_pv)]

qs = np.zeros((len_recording - start_point, len(v_range_pv), len(distances)-1))
qs_repeats = np.zeros(qs.shape)
qs_repeats.resize(qs_repeats.shape + (10,), refcheck=False)

# Generate Q matrix

for i in range(repeats):
    for j in range(len(distances)-1):
        positions = [distances[j], distances[j+1]]
        qs_repeats[:, :, j, i] = generate_qs(len(c1_repeats[:, i]), positions, v_range_pv, fs_pv)

# Solve using Cummins solver
w_repeats = []
for i in range(repeats):
    w = two_cap(c1_repeats[:, i], c2_repeats[:, i], qs_repeats[:, :, 0, i], qs_repeats[:, :, 1, i])
    w_repeats.append(w)

# Plotting
# 1. Original signals
for i in range(repeats):
    plt.plot(c1_repeats[:, i], label=f'Signal at Channel {ch1}')
    plt.plot(c2_repeats[:, i], label=f'Signal at Channel {ch2}')
plt.ylabel(['Amplitude (mV)'])
plt.xlabel(['Samples'])
# plt.legend()
plt.show()

# 2. Distribution
plt.plot(v_range_pv, np.max(w_repeats, axis=0))
plt.ylabel(['% of fibres'])
plt.xlabel(['Velocity (m/s)'])
plt.show()

# 3. Post-convolved
ct1_repeats = []
ct2_repeats = []
for i in range(repeats):
    ct1_repeats.append(np.convolve(c1_repeats[:, i], np.dot(qs_repeats[:, :, 1, i], w_repeats[i])))
    ct2_repeats.append(np.convolve(c2_repeats[:, i], np.dot(qs_repeats[:, :, 0, i], w_repeats[i])))


for i in range(repeats):
    plt.plot(ct1_repeats[i], label=f'Post-conv signal at Channel {ch1}')
    plt.plot(ct2_repeats[i], label=f'Post-conv signal at Channel {ch2}')
plt.ylabel(['Amplitude (mV)'])
plt.xlabel(['Samples'])
plt.show()

# 4. SFAPs
