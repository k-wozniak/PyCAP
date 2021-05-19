import abc
import numpy as np

class InterferenceSource():
    """ Class where the interference is generated based on the base signal. """
    
    @abc.abstractmethod
    def add_interference(self, signal:  np.ndarray) -> np.ndarray:
        """ Return signal with added interference """
        raise NotImplementedError