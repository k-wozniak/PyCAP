from PyCAP.excitationSources.excitation_source import ExcitationSource
import numpy as np

from enum import Enum    

class AccurateExcitationSource(ExcitationSource):
    """ Generates identical SFAPs. for each CV.
    
    - Multiple sfap sources
    - Multiple creation delays
    - cv^2 relationship with voltage

    """

    SFAP: np.ndarray

    fs: int = 100e3

    base_voltage = 0
    micro = 10e-6

    # Activation delay
    '''
    Tb: float = 1
    tb: float = 1
    I: float = 1
    Ib: float = 1
    '''

    def __init__(self, time_series: np.ndarray):
        """ Set basic components. """

        self.start = 1
        self.is_continuous = False

        #self.SFAP = self.A * np.power(time_series, self.n) * np.exp(-self.B * time_series)
        self.SFAP = np.array([self.get_sfap_function(x) for x in time_series])

    def get_sfap(self, velocity: float, time_shift: int = 0) -> np.ndarray:
        """ Returns the same SFAP for each velocity """
        temp = np.pad(self.SFAP, (time_shift, 0), 'constant', constant_values=(self.base_voltage, self.base_voltage))
        temp = temp[:-time_shift]

        return temp * (velocity ** 2)

    def get_excitation_delay(self, cv: float) -> float:
        pass

    def get_sfap_function(self, t_ms):
        t = t_ms * 10e3

        if t <= 0 or t >= 4.4:
            out = self.base_voltage
        else:
            fx = lambda x : (-0.0419396104*(x**15) + 1.4927258073*(x**14) - 23.5214743134*(x**13)
                + 215.0760694124*(x**12) - 1252.5931511956*(x**11) + 4761.0829202938*(x**10)
                - 11343.7377609156*(x**9) + 13435.1218524251*(x**8) + 7485.2236507130*(x**7)
                - 57638.1709034768*(x**6) + 100870.7157785190*(x**5) - 91671.1735290352*(x**4)
                + 44432.3147612717*(x**3) - 10098.5953278389*(x**2) + 879.2386412865*x - 6.1742125873)

            out = fx(t) -10

        return (out * self.micro)