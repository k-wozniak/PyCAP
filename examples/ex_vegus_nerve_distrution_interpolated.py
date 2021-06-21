from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

#caps = loadmat("meanCAP2.mat")['d']
#caps = loadmat("meanCAP2scaled.mat")['out']
#caps = loadmat("CAP.mat")['d']

caps = loadmat("meanCAP_interpolated3.mat")['d']
caps = np.array(caps).T
caps = np.flip(caps, 0)

num_electrodes, signal_length = caps.shape

fs = 100e3 * 3 # Hz * 10 because of the interpolation
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
#diss = be.mean_two_cap(caps, qs)
#diss = be.NCap(caps, qs)
diss = be.NCapPairs(caps, qs)

diss = moving_average(diss, 5)
plt.bar(search_range, diss)
plt.show()

diss = diss / (search_range**2)

plt.bar(search_range, diss)
plt.show()

'''
fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.set_title("diss_mean_two_CAP")
ax1.bar(search_range, diss_mean_two_CAP)

ax2.set_title("diss_NCAP")
ax2.bar(search_range, diss_NCAP)

ax3.set_title("diff_NCAP_Pairs")
ax3.bar(search_range, diff_NCAP_Pairs)
'''
plt.show()