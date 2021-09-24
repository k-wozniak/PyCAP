from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
import PyCAP.solvers.utils.sfap_reconstruction as reconstruct
import PyCAP.solvers.utils.sfap_reconstruction as sfap_rec

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

#caps = loadmat("meanCAP.mat")['d']
signals = loadmat("data/meanCAP2.mat")['d']

signal_length, num_electrodes = signals.shape

signals = np.array(signals).T

fs = 100e3 # Hz

du = 3.5e-3
distance_first_electrode = 80e-3

resolution = 0.5
search_range = np.arange(15, 85, resolution)

qs = []
for n in range(num_electrodes + 1):
    position = distance_first_electrode + (n * du)
    q = generate_q(signal_length, position, search_range, fs)
    qs.append(q)

w = be.NCapPairs(signals, qs)
diss = w.copy() / (search_range**2)

plt.figure("Original w")
plt.bar(search_range, w)
plt.show(block=False)

plt.figure("W scaled v^2 (distribution)")
plt.bar(search_range, diss)
plt.show(block=False)

As = []
for i in range(len(signals)):
    As.append(sfap_rec.find_sfap_A(signals[i], qs[i+1] - qs[i], diss, 800))

A = np.mean(As, axis=0)

i = 2

plt.figure("SFAP shape")
plt.plot(A[:, 0])
plt.show(block=False)

plt.figure("Reconstructed Signal")
plt.plot(A@(qs[i+1]-qs[i])@diss)
plt.show(block=False)

plt.figure("Signal")
plt.plot(signals[i])
plt.show(block=False)

plt.show()
