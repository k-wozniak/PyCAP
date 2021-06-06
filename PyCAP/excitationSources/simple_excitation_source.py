from PyCAP.excitationSources.excitation_source import ExcitationSource
import numpy as np

class SimpleExcitationSource(ExcitationSource):
    """ Generates identical SFAPs. for each CV. """

    A: int = 40744
    B: int = 15e+3
    n: int = 1

    SFAP: np.ndarray

    def __init__(self, time_series: np.ndarray):
        """ Set basic components. """

        self.start = 1
        self.is_continuous = False

        self.SFAP = self.A * np.power(time_series, self.n) * np.exp(-self.B * time_series)

    def get_sfap(self, velocity: float, time_shift: int = 0) -> np.ndarray:
        """ Returns the same SFAP for each velocity """
        #np.roll(self.SFAP, time_shift)

        temp = np.pad(self.SFAP, (time_shift, 0))
        temp = temp if time_shift == 0 else temp[:-time_shift]

        return temp