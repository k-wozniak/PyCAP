import numpy as np
import matplotlib.pyplot as plt

def get_sfap(t):
    t = t * 10e3

    if t <= 0 or t >= 4.4:
        return -70

    fx = lambda x : (-0.0419396104*(x**15) + 1.4927258073*(x**14) - 23.5214743134*(x**13)
        + 215.0760694124*(x**12) - 1252.5931511956*(x**11) + 4761.0829202938*(x**10)
        - 11343.7377609156*(x**9) + 13435.1218524251*(x**8) + 7485.2236507130*(x**7)
        - 57638.1709034768*(x**6) + 100870.7157785190*(x**5) - 91671.1735290352*(x**4)
        + 44432.3147612717*(x**3) - 10098.5953278389*(x**2) + 879.2386412865*x - 76.1742125873)

    out = fx(t)

    return out

fs = 100e3
simulation_length = 0.001

time_series = np.arange(0, simulation_length, 1/fs)

sfap = np.array([get_sfap(x) for x in time_series])

plt.plot(sfap)
plt.show()