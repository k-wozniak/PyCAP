from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import operator

def generate_qs(electrode_spacings, signal_length, search_range, fs):
    qs = []
    for position in electrode_spacings:
        q = generate_q(signal_length, position, search_range, fs)
        qs.append(q)

    return qs

def sum_of_squared_diff(old, new):
    diff = old - new
    return np.sum(diff**2)

def normalise_cv_diss_sq(w, search_range_cvs):
    for i in range(len(w)):
        v = search_range_cvs[i]
        w[i] = w[i] / (v**2)
    return w

def find_median_distribution(ws_median, resolution):
    ws_median = np.array(ws_median)
    ws_median = np.median(ws_median, axis=0) # Find the middle value of each velocity, not effected by outliers as much as mean
    ws_median[-int(10/resolution):] = 0 # Last values are mostly due to noise and general errors so set them to zero
    ws_median = ws_median / np.sum(ws_median) # Normalise as some values were removed

    return ws_median

def ssd_for_each_pair(dictionary, w_median):
    SSDs = []

    for key, value in dictionary.items():
        s = sum_of_squared_diff(w_median, value)
        SSDs.append((s, key, value))

    sort_f = lambda SSD : SSD[0]
    SSDs.sort(key=sort_f)

    return SSDs

def get_individual_electrode_sum_ssd(SSDs): # (ssd, (a, b), w)
    individual_ssds = dict()
    
    for s in SSDs:
        for k in s[1]:
            individual_ssds[k] = individual_ssds.get(k, 0) + s[0]
    
    return individual_ssds

# Load data and setup basic info ----------------------------------------------
caps = loadmat("meanCAP2.mat")['d']


samples = 1
signal_length, num_signals = caps.shape

caps = np.array(caps).T
caps = np.flip(caps, 0) # Should they be flipped?

fig, ax = plt.subplots(9)
for i in range(num_signals):
    ax[i].plot(caps[i])
plt.show()

# Set information about the signal and search range ---------------------------
fs = 100e3 # Hz
du = 3.5e-3
distance_first_electrode = 80e-3
resolution = 0.25
search_range = np.arange(10, 120, resolution)

# Initial electrode spacing
electrode_spacings = (np.array(range(num_signals+1)) * du) + distance_first_electrode
# electrode_spacings = [0.08, 0.0835, 0.0870, 0.0905, 0.09095652173913044, 0.0975, 0.101, 0.1045, 0.108, 0.1115]

# generate Qs -----------------------------------------------------------------
qs = generate_qs(electrode_spacings, signal_length, search_range, fs)

# Run algorithm ---------------------------------------------------------------
ws = {}
# Find 2CAP for each possible electrode pair
for i in range(num_signals - 1): # ignore last electrode
    for j in range(i+1, num_signals): # start at the next electrode
        w = be.two_cap(caps[i], caps[j], qs[i], qs[i+1], qs[j], qs[j+1])
        #w = normalise_cv_diss_sq(w, search_range)
        ws[(i, j)] = w

w_median = find_median_distribution([w for w in ws.values()], resolution)
SSDs = ssd_for_each_pair(ws, w_median)
individual_ssds = get_individual_electrode_sum_ssd(SSDs)

worst_electrode = max(individual_ssds.items(), key=operator.itemgetter(1))
worst_electrode_num = worst_electrode[0]
second_best_ssd = sorted(individual_ssds.values(), reverse=True)[1]
d_pos = (du/1.15, electrode_spacings[worst_electrode_num])

stagnation_exit = (0.00001, 1) # (aimed, current)
max_repeats = (100, 0) # (max, current)
current_best_ssd = worst_electrode[1]

steps_number = 10

while (stagnation_exit[0] < stagnation_exit[1] and 
        max_repeats[0] > max_repeats[1] and 
        second_best_ssd < current_best_ssd):
    
    # Generate new positions
    spacing = (2*d_pos[0]) / steps_number
    search_values = np.arange(d_pos[1] - d_pos[0], d_pos[0] + d_pos[1] + (0.5*spacing), spacing)
    temp_qs = generate_qs(search_values, signal_length, search_range, fs)

    new_ssds = {}
    for pos in range(len(search_values)): 
        new_ssd = 0
        for i in [x for x in range(num_signals) if x != worst_electrode_num]: # What about if its last electrode?
            qi2 = qs[i+1]
            if i+1 == worst_electrode_num:
                qi2 = temp_qs[pos]

            w = be.two_cap(caps[i], caps[worst_electrode_num], qs[i], qi2, temp_qs[pos], qs[worst_electrode_num+1])
            new_ssd = new_ssd + sum_of_squared_diff(w_median, w)

        new_ssds[pos] = new_ssd

    # choose the best ssd
    best_position = min(new_ssds.items(), key=operator.itemgetter(1))
    
    # update the new best position

    # update the search range
    d_pos = (spacing, search_values[best_position[0]])
    
    # update loop variables
    max_repeats = (max_repeats[0], max_repeats[1] + 1)
    stagnation_exit = (stagnation_exit[0], abs(current_best_ssd - best_position[1]))
    current_best_ssd = best_position[1]


a = 1
"""
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
    
    temp_qs = generate_qs(search_values, signal_length, search_range, fs)

    # find ssd for each new position
    

    for i in range(num_signals - 1): # ignore last electrode
        for j in range(i+1, num_signals): # start at the next electrode
            w = be.two_cap(caps[i], caps[j], qs[i], qs[i+1], qs[j], qs[j+1])

            for k in range(len(w)):
                v = search_range[i]
                w[i] = w[i] / (v**2)

            ws[(i, j)] = w


    # choose the best ssd

    # update the new best position

    # update the search range

    # 

    break

a = a
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