from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Load data and setup basic info ----------------------------------------------
caps = loadmat("meanCAP2.mat")['d']

samples = 1
signal_length, num_electrodes = caps.shape

caps = np.array(caps).T
#caps = np.fliplr(caps) # Should they be flipped?

# Set information about the signal and search range ---------------------------
fs = 100e3 # Hz
du = 3.5e-3
distance_first_electrode = 80e-3
search_range = np.arange(10, 120, 0.1)

# generate Qs -----------------------------------------------------------------
qs = []
for n in range(num_electrodes + 1):
    position = distance_first_electrode + (n * du)
    q = generate_q(signal_length, position, search_range, fs)
    qs.append(q)

# Run algorithm ---------------------------------------------------------------
end_offset = 100

ws = {}
# Find 2CAP for each possible electrode pair
for i in range(num_electrodes - 1):
    for j in range(i+1, num_electrodes):
        w = be.two_cap(caps[i], caps[j], qs[i], qs[i+1], qs[j], qs[j+1])

        #for k in range(len(w)):
        #    v = search_range[k]
        #    w[k] = w[k] / (v**2)

        w = w[0:-end_offset]

        # Normalise the distribution
        w = w / np.sum(w)

        ws[(i, j)] = w

# Find sum of squares for each difference
SSDs = []
ignore_list = []
for key_base, value_base in ws.items():
    ignore_list.append(key_base)

    for key, value in ws.items():
        if key not in ignore_list:
            diff = value_base - value
            s = np.sum(diff**2)

            SSDs.append((s, key_base, key))

sort_f = lambda SSD : SSD[0]
SSDs.sort(key=sort_f)

print("Smallest SSD is " + str(SSDs[1][0]) + " Pair " + str(SSDs[1][1]) + " " + str(SSDs[1][2]))
print("Largest SSD is " + str(SSDs[-1][0]) + " Pair " + str(SSDs[-1][1]) + " " + str(SSDs[-1][2]))

# Plot output -----------------------------------------------------------------

fig, ax = plt.subplots(2, 2)

ax[0, 0].set_title("Smallest SSD 1")
ax[0, 0].bar(search_range[0:-end_offset], ws[SSDs[1][1]])

ax[0, 1].set_title("Smallest SSD 2")
ax[0, 1].bar(search_range[0:-end_offset], ws[SSDs[1][2]])

ax[1, 0].set_title("Largest SSD 1")
ax[1, 0].bar(search_range[0:-end_offset], ws[SSDs[-1][1]])

ax[1, 1].set_title("Largest SSD 2")
ax[1, 1].bar(search_range[0:-end_offset], ws[SSDs[-1][2]])

plt.show()


fig, ax = plt.subplots(4)

ax[0].set_title("Signal " + str(SSDs[-1][1][0]))
ax[0].plot(caps[SSDs[-1][1][0]])

ax[1].set_title("Signal " + str(SSDs[-1][1][1]))
ax[1].plot(caps[SSDs[-1][1][1]])

ax[2].set_title("Signal " + str(SSDs[-1][2][0]))
ax[2].plot(caps[SSDs[-1][2][0]])

ax[3].set_title("Signal " + str(SSDs[-1][2][1]))
ax[3].plot(caps[SSDs[-1][2][1]])

plt.show()