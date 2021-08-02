from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


t = np.linspace(0, 1, 1500, endpoint=False)
s = signal.square(2 * np.pi * t * 5, 0.05)

plt.plot(t, s)
plt.show()