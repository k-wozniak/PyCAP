from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def generate_qs(electrode_spacings, signal_length, search_range, fs):
    qs = []
    for position in electrode_spacings:
        q = generate_q(signal_length, position, search_range, fs)
        qs.append(q)

    return qs

# Load data and setup basic info ----------------------------------------------
caps = loadmat("meanCAP2.mat")['d']

samples = 1
signal_length, num_signals = caps.shape

caps = np.array(caps).T
#caps = np.fliplr(caps) # Should they be flipped?

# Set information about the signal and search range ---------------------------
fs = 100e3 # Hz
du = 3.5e-3
distance_first_electrode = 80e-3
resolution = 0.25
search_range = np.arange(10, 120, resolution)

# Initial electrode spacing
electrode_spacings = (np.array(range(num_signals+1)) * du) + distance_first_electrode

# generate Qs -----------------------------------------------------------------
qs = generate_qs(electrode_spacings, signal_length, search_range, fs)

# Run algorithm ---------------------------------------------------------------
ws = {}
# Find 2CAP for each possible electrode pair
for i in range(num_signals - 1): # ignore last electrode
    for j in range(i+1, num_signals): # start at the next electrode
        w = be.two_cap(caps[i], caps[j], qs[i], qs[i+1], qs[j], qs[j+1])

        for k in range(len(w)):
            v = search_range[i]
            w[i] = w[i] / (v**2)

        ws[(i, j)] = w

ws_median = [w for w in ws.values()]
ws_median = np.array(ws_median)
ws_median = np.median(ws_median, axis=0) # Find the middle value of each velocity, not effected by outliers as much as mean
ws_median[-int(10/resolution):] = 0 # Last values are mostly due to noise and general errors so set them to zero
ws_median = ws_median / np.sum(ws_median) # Normalise as some values were removed

# Find sum of squares for each difference
SSDs = []
SSDs_sum = np.zeros(num_signals)
for key, value in ws.items():
    diff = ws_median - value
    s = np.sum(diff**2)

    SSDs_sum[key[0]] = SSDs_sum[key[0]] + s
    SSDs_sum[key[1]] = SSDs_sum[key[1]] + s

    SSDs.append((s, key, value))

sort_f = lambda SSD : SSD[0]
SSDs.sort(key=sort_f)

print("Smallest SSD is " + str(SSDs[1][0]) + " For " + str(SSDs[1][1]))
print("Largest SSD is " + str(SSDs[-1][0]) + " For " + str(SSDs[-1][1]))

index = np.argmax(SSDs_sum)
value_original = SSDs_sum[index]
value_second = sorted(SSDs_sum)[-2]

original_position = electrode_spacings[index]
new_best_position = original_position

current_search_range = (original_position-du, original_position+du)

steps_number = 10
new_value = value_original
while new_value > value_second: # Perform optimisation
    # Generate shifts to check
    spacing = (current_search_range[1] - current_search_range[0]) / steps_number
    search_values = np.arange(current_search_range[0], current_search_range[1] + (0.5*spacing), spacing)
    
    


    break


# Plot output -----------------------------------------------------------------

fig, ax = plt.subplots(2, 2)

ax[0, 0].set_title("Smallest SSD 1")
ax[0, 0].bar(search_range, SSDs[1][2])

ax[0, 1].set_title("Smallest SSD 2")
ax[0, 1].bar(search_range, SSDs[2][2])

ax[1, 0].set_title("Largest SSD -2")
ax[1, 0].bar(search_range, SSDs[-2][2])

ax[1, 1].set_title("Largest SSD -1")
ax[1, 1].bar(search_range, SSDs[-1][2])

plt.show()

"""
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
"""