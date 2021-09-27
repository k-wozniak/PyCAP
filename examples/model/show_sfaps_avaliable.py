import PyCAP.excitationSources as sources
import matplotlib.pyplot as plt
import numpy as np

simulation_length = 0.001
fs = 400e3
time_series = np.arange(0, simulation_length, 1/fs)

# Plot singular signals obtained
fig, ax = plt.subplots(3)

ax[0].set_title("Simple excitation source")
ax[0].set_xlabel("Time s")
ax[0].set_ylabel("Amplitude")
ax[0].plot(sources.SimpleExcitationSource(time_series).get_sfap(25))

ax[1].set_title("Accurate excitation source")
ax[1].set_xlabel("Time s")
ax[1].set_ylabel("Amplitude")
ax[1].plot(sources.AccurateExcitationSource(time_series).get_sfap(25))

ax[2].set_title("PyPNS like excitation source")
ax[2].set_xlabel("Time s")
ax[2].set_ylabel("Amplitude")
ax[2].plot(sources.PyPnsLike(time_series).get_sfap(25))

plt.show()