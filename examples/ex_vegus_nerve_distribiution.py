from PyCAP.solvers import bipolar_electrodes as be
from PyCAP.solvers.utils.qs_generation import generate_q
from PyCAP.solvers.utils.signal_operations import moving_average

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

caps = loadmat("meanCAP2.mat")['d']
#caps = loadmat("CAP.mat")['d']

samples = 1
signal_length, num_electrodes = caps.shape
#signal_length, num_electrodes, samples = caps.shape

caps = np.array(caps).T
#caps = np.fliplr(caps)

fs = 100e3 # Hz

du = 3.5e-3
distance_first_electrode = 80e-3

for resolution in [0.1]:
    search_range = np.arange(10, 120, resolution)

    qs = []
    for n in range(num_electrodes + 1):
        position = distance_first_electrode + (n * du)
        q = generate_q(signal_length, position, search_range, fs)
        qs.append(q)
    
    ws = []
    ws_lin = []
    ws_quad = []

    for s in range(samples): # samples):
        #signals = (caps[s, :, :])
        signals = caps
        w = be.NCap(signals, qs)

        w_lin = w.copy()
        w_quad = w.copy()

        for i in range(len(w)):
            v = search_range[i]

            w_lin[i] = w[i] / v
            w_quad[i] = w[i] / (v**2)

        ws.append(w)
        ws_lin.append(w_lin)
        ws_quad.append(w_quad)

    #plt.plot(np.mean(ws, axis=0))
    #plt.plot(np.mean(ws_lin, axis=0))
    #plt.plot(search_range, ws_quad)
    #search_range = np.arange(10, 80, 0.25)
    #search_range = search_range[1:-1]
    plt.bar(search_range, ws_quad[0])
    plt.show()

    plt.bar(search_range, ws_quad[0])
    plt.bar(search_range, moving_average(ws_quad[0], 3))
    plt.show()

    #plt.plot(search_range, ws[0])
    #plt.bar(search_range, moving_average(ws[0], 3))
    #plt.show()

    file_name = "w_distribution" + str(resolution) + ".mat"
    savemat(file_name, {"ws": ws, "ws_lin": ws_lin, "w_quad": ws_quad})

