from numpy.core.shape_base import block
from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import interpolate_signals
import PyCAP.solvers.utils.sfap_reconstruction as sfap_rec
from PyCAP.io.load import StimulationData
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

stimulation_data = StimulationData(loadmat("../data/electricalStimulation.mat"))

signals_dataset = []
for lv in [0]: #, 49, 48, 47, 46]:
    signals_dataset.extend(stimulation_data.get_signals(lv, 51))

signals_dataset = [interpolate_signals(sig, 5, 100e3) for sig in signals_dataset]

fs = 100e3 * 5 # Hz interpolated
du = 3.5e-3
distance_first_electrode = 80e-3
electrode_positions = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * du) + distance_first_electrode

# Search definitions
resolution = 1
min_cv = 10
max_cv = 80
search_range = np.arange(min_cv, max_cv, resolution)

# Pre-calculate qs as they do not change between runs
qs = []
signal_length = signals_dataset[0][0].shape[0]
for pos in electrode_positions:
    q = generate_q(signal_length, pos, search_range, fs)
    qs.append(q)

# Search ----------------------------------------------------------------------
#diss_mean_two_CAP = []
#diss_NCAP = []
diss_NCAP_Pairs = []
#diss_VSR = []

As = []
for signals in signals_dataset:
    #diss_mean_two_CAP.append(be.mean_two_cap(signals, qs))
    #diss_NCAP.append(be.NCap(signals, qs))
    #diss_NCAP_Pairs.append(be.NCapPairs(signals, qs))
    #diss_VSR.append(be.VSR(signals, fs, du, min_cv, resolution, max_cv+1))
    
    w = be.NCapPairs(signals, qs)
    diss_NCAP_Pairs.append(w)
    
    for i in range(len(signals)):
        As.append(sfap_rec.find_sfap_A(signals[i], qs[i+1]-qs[i], w))
"""
to_save = {
    "mean_two_cap": diss_mean_two_CAP,
    "ncap": diss_NCAP,
    "ncap_pairs": diss_NCAP_Pairs,
    "VSR": diss_VSR,
}
"""

mean_diss = np.mean(diss_NCAP_Pairs, axis=0)
std = np.std(diss_NCAP_Pairs, axis=0)

plt.figure()
plt.bar(search_range, mean_diss, yerr=std)
plt.show(block=False)



A = np.mean(As, axis=0)

plt.figure()
plt.plot(A[:, 0])
plt.show(block=False)

i = 2
plt.figure()
plt.plot((A@(qs[i+1]-qs[i])@w))
plt.plot(signals[i])
plt.show(block=False)

plt.show()
#savemat("disstributions-47-51.mat", to_save)
