from dataclasses import dataclass
import numpy as np

@dataclass(init=False)
class ModelParams():
    """ ModelParams is a data class which stores all information needed for the
        model. """

    # Sampling Frequency in Hz (default 100kHz in constructor)
    fs: int

    # CV distribution in m/s
    # Numpy matrix with first column cv, second number of fibres
    # Example [5m/s, 10m/s, 20m/s, 50m/s]
    #         [100,  50,    40,    2    ]
    cv_distribution: None

    # Simulation length (s)
    simulation_length: float = 0.030 #(s)
    time_series = []

    def __init__(self, cv_distribution, simulation_length: float, fs: int = 100_000):
        self.cv_distribution = cv_distribution
        self.simulation_length = simulation_length
        self.fs = fs
        self.time_series = np.arange(0, simulation_length, 1/fs)

    def number_of_cv_classes(self) -> int:
        return self.cv_distribution.shape[0]

    def get_cv_distribution(self) -> np.ndarray:
        return self.cv_distribution