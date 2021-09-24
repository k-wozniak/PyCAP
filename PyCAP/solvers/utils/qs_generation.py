from PyCAP.recordingProbes.recording_probe import RecordingProbe
from PyCAP.solvers.recorded_signals import RecordedSignals
import numpy as np
from typing import List
from numba import jit

""" Helper functions for generation Q matrices while generating the problem 
    space.
"""

#@jit(nopython=True)
def generate_q(signal_length: int, position: float, velocities: np.ndarray, fs: int):
    """ Generates Q matrix at the given position for the velocities passed at 
        the given frequency and of the signal length passad """
    v_length = velocities.shape[0]

    q = np.zeros((signal_length, v_length))

    for i in range(0, v_length):
        dt = int(round((position / velocities[i]) * fs))

        if dt <= signal_length and dt >= 0:
            q[dt, i] = 1

    return q

def generate_q_from_probe(recording_probe: RecordingProbe, velocities: np.ndarray, fs: int):
    """ Converts recording probe wrapper to used the base function """
    if not recording_probe.is_output_set():
        raise ValueError
    
    signal_length = recording_probe.output_signal.shape[0]
    position = recording_probe.position

    return generate_q(signal_length, position, velocities, fs)

def generate_qs_from_probes(recording_probes: List[RecordingProbe], velocities: np.ndarray, fs: int):
    """ Used generate qs from a list as it is a common procedure) """
    qs = []

    for recording_probe in recording_probes:
        qs.append(generate_q_from_probe(recording_probe, velocities, fs))

    return qs

def generate_qs_from_signals(recorded_signals: RecordedSignals, velocities: np.ndarray):
    """ Used to generate qs from a list of signals """

    qs = []

    for position, signal in recorded_signals.data.items():
        qs.append(generate_q(len(signal), position, velocities, recorded_signals.fs))

    return qs 