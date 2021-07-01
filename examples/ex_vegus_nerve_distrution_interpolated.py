from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average
from PyCAP.solvers.utils.sfap_reconstruction import find_matrix_A, recreate_A_matrix

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

#caps = loadmat("meanCAP2.mat")['d']
#caps = loadmat("meanCAP2scaled.mat")['out']
#caps = loadmat("CAP.mat")['d']
# _interpolated10
caps = loadmat("meanCAP.mat")['d']
caps = np.array(caps).T
caps = np.flip(caps, 0)

num_electrodes, signal_length = caps.shape

fs = 100e3  # Hz * 10 because of the interpolation
du = 3.5e-3
distance_first_electrode = 80e-3

#caps = np.delete(caps, (4), axis=0)
electrode_positions = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * du) + distance_first_electrode

# Search definitions
resolution = 1
min_cv = 10
max_cv = 75
search_range = np.arange(min_cv, max_cv, resolution)

# Pre-calculate qs as they do not change between runs
qs = []
for pos in electrode_positions:
    q = generate_q(signal_length, pos, search_range, fs)
    qs.append(q)


# Methods used to determine distributions
diss_mean = be.mean_two_cap(caps, qs)
diss_ncap = be.NCap(caps, qs)
diss_pairs = be.NCapPairs(caps, qs)
diss_VSR = be.VSR(caps, fs, du, min_cv, resolution, max_cv)

diss_mean[diss_mean < 0] = 0
diss_ncap[diss_ncap < 0] = 0
diss_pairs[diss_pairs < 0] = 0
diss_VSR[diss_VSR < 0] = 0

diss_pairs =  diss_pairs / (search_range**2)
diss_pairs = moving_average(diss_pairs, 5)
diss_pairs = diss_pairs / np.sum(diss_pairs)

diss_pairs = diss_pairs * 1e6
diss_pairs = np.round(diss_pairs)
"""
plt.bar(search_range, diss_VSR)
plt.show()
"""
print(diss_pairs)

A = find_matrix_A(caps[0, :], qs[0], diss_pairs) * 100

AA = recreate_A_matrix(A)

idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
a2 = np.delete(A, idx, axis=1)

plt.plot(A[:, 108])
plt.show()

plt.plot(AA[:, 0])
plt.show()


# Try to scale
title = "w. Resolution: " + str(resolution) + "Range: " + str(min_cv) + " - " + str(max_cv)
plt.suptitle(title) #, fontsize=16)

plt.subplot(141)
plt.title("diss mean two CAP")
plt.bar(search_range, diss_mean)

plt.subplot(142)
plt.title("diss NCAP")
plt.bar(search_range, diss_ncap)

plt.subplot(143)
plt.title("diff NCAP Pairs")
plt.bar(search_range, diss_pairs)

plt.subplot(144)
plt.title("diff VSR")
plt.bar(search_range, diss_VSR)

plt.show()

title = "w. (Moving Average 5) Resolution: " + str(resolution) + "Range: " + str(min_cv) + " - " + str(max_cv)
plt.suptitle(title) #, fontsize=16)

plt.subplot(141)
plt.title("diss mean two CAP")
plt.bar(search_range, moving_average(diss_mean, 5))

plt.subplot(142)
plt.title("diss NCAP")
plt.bar(search_range, moving_average(diss_ncap, 5))

plt.subplot(143)
plt.title("diff NCAP Pairs")
plt.bar(search_range, moving_average(diss_pairs, 5))

plt.subplot(144)
plt.title("diff VSR")
plt.bar(search_range, moving_average(diss_VSR, 5))

plt.show()

diss_mean = diss_mean / (search_range**2)
diss_mean = moving_average(diss_mean, 5)

diss_ncap = diss_ncap / (search_range**2)
diss_ncap = moving_average(diss_ncap, 5)

diss_pairs = diss_pairs / (search_range**2)
diss_pairs = moving_average(diss_pairs, 5)

diss_VSR = diss_VSR / (search_range**2)



title = "Normalised. Resolution: " + str(resolution) + "Range: " + str(min_cv) + " - " + str(max_cv)
plt.suptitle(title) #, fontsize=16)

plt.subplot(141)
plt.title("diss mean two CAP")
plt.bar(search_range, diss_mean)

plt.subplot(142)
plt.title("diss NCAP")
plt.bar(search_range, diss_ncap)

plt.subplot(143)
plt.title("diff NCAP Pairs")
plt.bar(search_range, diss_pairs)

plt.subplot(144)
plt.title("diff VSR")
plt.bar(search_range, diss_VSR)

plt.show()