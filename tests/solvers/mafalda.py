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

signals = caps_pv[:, :, 0]

ch1 = 1
ch2 = 2

# Single repeat only:
c1 = signals[100:, ch1]
c2 = signals[100:, ch2]
c1 = c1 - np.mean(c1)
c2 = c2 - np.mean(c2)

# Distances
distances = [80e-3, 80e-3 + (ch1 * du_pv), 80e-3 + (ch2 * du_pv)]
qs = np.zeros((len(c1), len(v_range_pv), len(distances)-1))

# Generate Q matrix
for i in range(len(distances)-1):
    positions = [distances[i], distances[i+1]]
    qs[:, :, i] = generate_qs(len(c1), positions, v_range_pv, fs_pv)

# Solve using Cummins solver
w = two_cap(c1, c2, qs[:, :, 0], qs[:, :, 1])

# Plotting
# 1. Original signals
plt.plot(c1, label=f'Signal at Channel {ch1}')
plt.plot(c2, label=f'Signal at Channel {ch2}')
plt.ylabel(['Amplitude (mV)'])
plt.xlabel(['Samples'])
plt.legend()
plt.show()

# 2. Distribution
plt.plot(v_range_pv, w)
plt.ylabel(['% of fibres'])
plt.xlabel(['Velocity (m/s)'])
plt.show()

# 3. Post-convolved
ct1 = np.convolve(c1, np.dot(qs[:, :, 1], w))
ct2 = np.convolve(c2, np.dot(qs[:, :, 0], w))

plt.plot(ct1, label=f'Post-conv signal at Channel {ch1}')
plt.plot(ct2, label=f'Post-conv signal at Channel {ch2}')
plt.ylabel(['Amplitude (mV)'])
plt.xlabel(['Samples'])
plt.legend()
plt.show()

# 4. SFAPs
