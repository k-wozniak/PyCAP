import abc
import numpy as np

class RecordingProbe():
    """ RecordingProbe class is the based class for each recording mechanisms
    intended to be added to the model. It contains all required fields and 
    functions. It is kept as simple as possible to allow greate leavel of 
    flexibility when implementing different interfaces, such as cuff 
    electrodes. """

    position: float = 0.08 # Position along the nerve bundle in meters (m) 
    output_signal: np.ndarray = None # The computed output signal

    def get_position(self) -> float:
        """ Returns the position of the recording probe """
        return self.position

    def is_output_set(self) -> bool:
        """ Returns information if the output is avaliable """
        return (self.output_signal is not None)

    @abc.abstractmethod
    def set_output_signal(self, signal: np.ndarray) -> None:
        """ Used to set the output signal as in case of different electrodes
            Noise and specific characteristics may be added """
        raise NotImplementedError