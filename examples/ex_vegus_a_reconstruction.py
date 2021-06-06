from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
import PyCAP.solvers.utils.sfap_reconstruction as reconstruct

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

#caps = loadmat("meanCAP.mat")['d']
signals = loadmat("meanCAP2.mat")['d']

signal_length, num_electrodes = signals.shape

signals = np.array(signals).T

fs = 100e3 # Hz

du = 3.5e-3
distance_first_electrode = 80e-3

resolution = 0.25
search_range = np.arange(10, 120, resolution)

qs = []
for n in range(num_electrodes + 1):
    position = distance_first_electrode + (n * du)
    q = generate_q(signal_length, position, search_range, fs)
    qs.append(q)

bipolar_qs = []
for i in range(len(qs)-1):
    bipolar_qs.append( qs[i+1] - qs[i] )

w = be.NCap(signals, qs)

w_lin = w.copy()
w_quad = w.copy()

for i in range(len(w)):
    v = search_range[i]
    w_lin[i] = w[i] / v
    w_quad[i] = w[i] / (v**2)

#plt.plot(np.mean(ws, axis=0))
#plt.plot(np.mean(ws_lin, axis=0))
#plt.plot(search_range, w_quad)
#plt.show()

A = reconstruct.find_matrix_A_from_set(signals, bipolar_qs, w_quad)
A2 = reconstruct.recreate_A_matrix(A.T).T # does not work

AQ = np.matmul(A, np.asmatrix(bipolar_qs[0]))
reconstructed_signal = np.matmul(AQ, w_quad)
'''
fig, axs = plt.subplots(3)
axs[0].set_title("Original meanCAP2 signal")
axs[0].plot(signals[0])
axs[1].set_title("Reconstructed meanCAP2 signal")
axs[1].plot(reconstructed_signal.T)
axs[2].set_title("Found distribution")
axs[2].plot(search_range, w_quad)
plt.show()
'''
y = range(0, len(A))

fig, axs = plt.subplots(2)
axs[0].set_title("A calculated")
axs[0].plot(y, A)
axs[1].set_title("A reconstructed")
axs[1].plot(y, A2)

plt.show()
#file_name = "correct_output" + str(resolution) + ".mat"
#savemat(file_name, {"ws": ws, "ws_lin": ws_lin, "w_quad": ws_quad})