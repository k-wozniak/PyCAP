from PyCAP.recordingProbes.recording_probe import RecordingProbe
import numpy as np

""" Helper functions for generation Q matrices while generating the problem 
    space.
"""

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

def generate_q_matrix(recording_probe: RecordingProbe, velocities: np.ndarray, fs: int):
    """ Converts recording probe wrapper to used the base function """
    if not recording_probe.is_output_set():
        raise ValueError
    
    signal_length = recording_probe.output_signal.shape[0]
    position = recording_probe.position

    return generate_q(signal_length, position, velocities, fs)

def generate_qs(recording_probes, velocities: np.ndarray, fs: int):
    """ Used generate qs from a list as it is a common procedure) """
    qs = []

    for recording_probe in recording_probes:
        qs.append(generate_q_matrix(recording_probe, velocities, fs))

    return qs

