from PyCAP.recordingProbes.recording_probe import RecordingProbe
from PyCAP.excitationSources.excitation_source import ExcitationSource
from PyCAP.interferenceSources.interference_source import InterferenceSource
from PyCAP.model.model_params import ModelParams

from collections.abc import Sized, Iterable, Iterator
import numpy as np

class Model():
    """ Model class is responsible for simulation of the peripheral nervous
        system as conceptualised by Cummins et al. (1978).
        The model consists of three main parts:
        1. Stimulation sources - provide CV dependent SFAP sources
        2. Interference sources - generate noise based on the signal
        3. Recording probes - used to not the location the probe is present and
            add electrode dependent behaviour if needed.
    """

    params: ModelParams = None

    excitation_sources = []
    recording_probes = []
    interference_sources = []

    def __init__(self, params: ModelParams):
        self.excitation_sources = []
        self.recording_probes = []
        self.interference_sources = []

        self.params = params

    def simulate(self: str) -> None:
        """ Iterates through each probe calculating the output signal with 
            Added interference """

        for probe in self.recording_probes:
            signal = self.find_probe_signal(probe)
            signal = self.add_interference(signal)
            probe.set_output_signal(signal)

    def find_probe_signal(self, probe: RecordingProbe) -> np.ndarray:
        """ Shifts each of the signals and accumulates it based on the number
            of the fibres in the class
        
        params : ModelParams    Parameters used to run the Model
        """
        signal = np.zeros(len(self.params.time_series))

        for cv in self.params.get_cv_distribution():
            for source in self.excitation_sources:
                velocity = cv[0]
                fibre_count = int(cv[1])

                distance = probe.get_position() - source.get_position()
                dt = int(round((distance/velocity)*self.params.fs))
                
                signal = signal + (source.get_sfap(velocity, dt) * fibre_count)
        
        return signal

    def add_interference(self, signal) -> np.ndarray:
        for interference in self.interference_sources:
            signal = interference.add_interference(signal)

        return signal

    def add_excitation_source(self, excitation_source: ExcitationSource) -> None:
        self.excitation_sources.append(excitation_source)

    def add_recording_probe(self, recording_probe: RecordingProbe) -> None:
        self.recording_probes.append(recording_probe)

    def add_interference_source(self, interference_source: InterferenceSource) -> None:
        self.interference_sources.append(interference_source)