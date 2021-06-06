import numpy as np
from typing import List

from PyCAP.recordingProbes.recording_probe import RecordingProbe

class OverlappingMultipolarElectrodes():
    """ OverlappingMultipolarElectrodes takes output electrodes after simulation
        and combines single signals into bi-polar / tri-polar etc recording 
        based on the setup parameter.

        Setup is an array of possitive and negative integers.
        The length of the array indicates electrode setup (bipolar for two values, 
        tripolar for three etc.). All values are adjusted to be equal to 1, 0, -1
        where 1 is addition, -1 subtraction and 0 ignore. For example setup = [-1, 1]

        Means a bipolar setup with following outputs
        Recording 0 -> electrode 1 - electrode 0
        Recording 1 -> electrode 2 - electrode 1
        Recording 2 -> electrode 3 - electrode 2
        
        etc.

        Recordings must be sorted!
    """
    def __init__(self, setup: List[int]):
        for s in setup:
            if s > 0:
                s = 1
            elif s < 0:
                s = -1
        
        if not isinstance(setup, list) or len(setup) < 1: # Must be minimum of 1
            raise ValueError

        self.setup = setup
        self.recording_probes = []

    def add_recording_probe(self, probe: RecordingProbe) -> None:
        self.recording_probes.append(probe)

    def add_recording_probes(self, probes: List[RecordingProbe]) -> None:
        for probe in probes:
            self.add_recording_probe(probe)

    def get_recording(self, position: int) -> np.ndarray:
        """ Position starts at 0.
        """
        # Check if enough electrodes for given setup
        if position < 0 or (position + len(self.setup)) > len(self.recording_probes):
            # Impossible to have multipolar electrode setup with given number of signals
            raise ValueError 

        # Check if every probe has set output
        for pos in range(position, (position + len(self.setup)), 1):
            if not self.recording_probes[pos].is_output_set():
                # Recording not present
                raise ValueError

        # Combine outputs into one signal
        signal = np.zeros(self.recording_probes[position].output_signal.shape)
        for i in range(len(self.setup)):
            signal = signal + (self.setup[i] * self.recording_probes[i+position].output_signal)

        return signal
            
    def get_all_recordings(self) -> np.ndarray:
        # Check if the first element has set array
        if not self.recording_probes[0].is_output_set():
            raise ValueError
        
        signals = []
        for pos in range(0, len(self.recording_probes) - len(self.setup) + 1, 1):
            signals.append(self.get_recording(pos))

        return np.asarray(signals)

