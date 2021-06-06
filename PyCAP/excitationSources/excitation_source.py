import abc
from enum import Enum
import numpy as np

class ExcitationSource(metaclass=abc.ABCMeta):
    """ ExcitationSource class is the based class for each excitation mechanisms
        intended to be added to the model. It contains all required fields and 
        functions. It is kept as simple as possible to allow greater level of 
        flexibility when implementing different interfaces, such as cuff 
        electrodes. """

    start: int = 0 # Defines the start of the first pulse
    is_continuous: bool = False # Defines if the signal repeats
    position: float = 0.00 # Position in meters (m)

    def get_position(self) -> float:
        """ Retruns stored position """
        return self.position

    @abc.abstractmethod
    def get_sfap(self, velocity: float, time_shift: int = 0) -> np.ndarray:
        """ Gets the sfap at the specific cv at given time shift """
        raise NotImplementedError