import numpy as np
import matplotlib.pyplot as plt
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers import bipolar_electrodes as be
from scipy.io import loadmat

path = u"C:/Users/admin/Documents/Projects/PyCAP/PyPNS_simulations/new/20_axons_1_pole_largeDist.mat"
data = loadmat(path)

SFAPs = data["SFAPs"]
time = data["time"]
signals_dataset = data["CAPs"]

nerve_vs = data['velocities'][0]

signals = []
for i in range(len(signals_dataset)-1):
    signals.append(signals_dataset[i+1] - signals_dataset[i])

signals = np.array(signals)

fs_compare = 1/((time[0][1]-time[0][0])*1e-3)
fs = 400e3
print(fs_compare)
print(fs)

du = 3.5e-3
distance_first_electrode = 80e-3
electrode_positions = (np.array([0, 1, 2, 3, 4]) * du) + distance_first_electrode

# Search definitions
resolution = 1
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

i = 1
j = 4
zero = np.zeros_like(qs[i])
w = be.two_cap(signals_dataset[i], signals_dataset[i+1], qs[i], zero, qs[j], zero)

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


"""
from scipy.io import savemat

file_name = "example"

cv_velocities = [20, 21, 22, 23] # ms-1
cv_distribution = [1, 5, 5, 1] # in units of nerves
fs = 100e3  # sampling frequency in Hz
probes_distances = [8000*1e-6, 12000*1e-6, 16000*1e-6, 20000*1e-6, 24000*1e-6] # in meters

signals = [
    [0, 1, 2, 3, 3, 1],
    [0, 0, 1, 2, 3, 3],
    [0, 0, 1, 2, 3, 3],
]

signals_sfap = [
    [0, 1, 2, 3, 3, 1],
    [0, 1, 2, 3, 3, 1],
    [0, 1, 2, 3, 3, 1],
]

savemat(file_name,
{   
    "cv_velocities": cv_velocities,
    "cv_diss": cv_distribution,
    "fs": fs,

    "probes_distances": probes_distances,
    
    "signals": signals,
    "signals_SFAP": signals_sfap,
})
"""
plt.show()