from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

def process_distributions(diss, resolution, target_resolution, min_cv, max_cv):
    # resolution = 0.1
    # target_resolution = 1
    padding = round(target_resolution/(2*resolution))

    # perform resolution scaling
    diss = np.sum(diss, 0)
    diss = np.pad(diss, (padding, padding), 'constant')
    
    mask = np.ones(padding*2)
    mask[0] = 0.5
    mask[-1] = 0.5

    diss = np.convolve(mask, diss, 'valid')
    diss = diss[::(padding*2)] # start:stop:step

    # perform velocity scaling (Should be replaced my matrix multiplication)
    new_cv_range = np.arange(min_cv, max_cv + (target_resolution/2), target_resolution)
    diss = diss / (new_cv_range**2)

    return diss

#caps = loadmat("meanCAP2.mat")['d']
#caps = loadmat("meanCAP2scaled.mat")['out']
#caps = loadmat("CAP.mat")['d']

data = loadmat("eCAPs_with_classes")

keys = np.array(data['keys'][0])
caps = np.array(data['samples'])

stimulation_levels = np.sort(np.unique(keys))[-5:]

num_electrodes, signal_length, _ = caps.shape

fs = 100e3 # Hz
du = 3.5e-3
distance_first_electrode = 80e-3

#caps = np.delete(caps, (4), axis=0)
electrode_positions = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * du) + distance_first_electrode

# Search definitions
resolution = 0.1
target_resolution = 1
min_cv = 10
max_cv = 120
search_range = np.arange(min_cv, max_cv, resolution)

# Pre-calculate qs as they do not change between runs
qs = []
for pos in electrode_positions:
    q = generate_q(signal_length, pos, search_range, fs)
    qs.append(q)

results = {}
for stimulation_level in stimulation_levels:
    print("Stimulation Level: " + str(stimulation_level))
    
    positions = [i for i, x in enumerate(keys) if x == stimulation_level]

    # Methods used to determine distributions
    diss_mean_two_CAP = []
    diss_NCAP = []
    diff_NCAP_Pairs = []
    diss_VSR = []


    for s in positions:
        signals = (caps[:, :, s])
        
        diss_mean_two_CAP.append(be.mean_two_cap(signals, qs))
        diss_NCAP.append(be.NCap(signals, qs))
        diff_NCAP_Pairs.append(be.NCapPairs(signals, qs))
        diss_VSR.append(be.VSR(signals, fs, du, min_cv, target_resolution, max_cv+1))
    

    # Process multiple distributions into one
    diss_mean_two_CAP = process_distributions(diss_mean_two_CAP, resolution, target_resolution, min_cv, max_cv)
    diss_NCAP = process_distributions(diss_NCAP, resolution, target_resolution, min_cv, max_cv)
    diff_NCAP_Pairs = process_distributions(diff_NCAP_Pairs, resolution, target_resolution, min_cv, max_cv)
    diss_VSR = np.mean(diss_VSR, axis=0)

    # Save results
    results[stimulation_level] = {
        "mean_two_cap": diss_mean_two_CAP,
        "ncap": diss_NCAP,
        "ncap_pairs": diff_NCAP_Pairs,
        "VSR": diss_VSR,
    }

items = results.items()
to_save = {
    "mean_two_cap": [value["mean_two_cap"] for _, value in items],
    "ncap": [value["ncap"] for _, value in items],
    "ncap_pairs": [value["ncap_pairs"] for _, value in items],
    "VSR": [value["VSR"] for _, value in items],
    "stimulations": [key for key, _ in items],
}

savemat("loads_of_data_to_sort.mat", to_save)