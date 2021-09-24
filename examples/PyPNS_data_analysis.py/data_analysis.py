import numpy as np
import matplotlib.pyplot as plt
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers import bipolar_electrodes as be
from scipy.io import loadmat
from PyCAP.analysis_utils import post_conv_analysis

path = u"C:/Users/admin/Documents/Projects/PyCAP/PyPNS_simulations/new/20_axons_1_pole_largeDist.mat"
data = loadmat(path)

SFAPs = data["SFAPs"]
time = data["time"]
signals_dataset = data["CAPs"]

nerve_vs = data['velocities'][0]
print(nerve_vs)

signals = []
for i in range(len(signals_dataset)-1):
    signals.append(signals_dataset[i+1] - signals_dataset[i])


signals = np.array(signals)

fs_compare = 1/((time[0][1]-time[0][0])*1e-3)
fs = 400e3
print(fs_compare)
print(fs)

du = 3.5e-2
distance_first_electrode = 80e-3
electrode_positions = (np.array([0, 1, 2, 3, 4]) * du) + distance_first_electrode

# Search definitions
resolution = 0.5
min_cv = 20
max_cv = 80
search_range = np.arange(min_cv, max_cv, resolution)

# Pre-calculate qs as they do not change between runs
qs = []
signal_length = signals_dataset[0].shape[0]
for pos in electrode_positions:
    q = generate_q(signal_length, pos, search_range, fs)
    qs.append(q)

# solve
w = be.NCapPairs(signals, qs)
ww = be.VSR(signals, fs, du, min_cv, resolution, max_cv)

#ost_conv_analysis(signals_dataset, qs, w, [(1, 3)])

plt.figure()
plt.title("W")
plt.bar(search_range, w)
plt.show(block=False)

plt.figure()
plt.title("W normalised")
plt.bar(search_range, (w / (search_range**2)))
plt.show(block=False)


plt.figure()
plt.title("Expected")
plt.bar(nerve_vs, np.ones(len(nerve_vs))/int(len(nerve_vs)))
plt.show(block=False)

plt.figure()
plt.title("VSR")
plt.plot(search_range, ww[1][:-1])
plt.show(block=False)


plt.show()