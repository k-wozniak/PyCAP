from PyCAP.recordingProbes.recording_probe import RecordingProbe
import numpy as np

class SimpleRecordingProbe(RecordingProbe):
    """ SimpleRecordingProbe is the most simple way of recording the output
        where no interference caused by the electrode are added (ideal case) """

    def __init__(self, position: float):
        self.position = position

    def set_output_signal(self, signal: np.ndarray) -> None:
        """ Simple set the output, does not effect the signal. """
        self.output_signal = signal