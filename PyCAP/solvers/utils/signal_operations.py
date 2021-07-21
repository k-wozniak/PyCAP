from scipy.linalg import toeplitz
from numpy.lib.function_base import interp
import numpy as np

def extend_signals_toeplitz(signals, qs_length):
    """ Extends a 1D signal array to a add each possible shift """
    extended_signals = []
    
    for s in signals:
        r = np.pad(s, (0, qs_length-1), 'constant')
        
        c = np.zeros(qs_length)
        c[0] = s[0]

        conv_sign = (toeplitz(c, r)).T

        extended_signals.append(conv_sign)

    return extended_signals

def moving_average(data_set, periods=3):
    """ Performs an moving average operation on the signal. Edges are extended 
        based on the current edge values. """
    extend = int(periods/2)
    data_set = np.pad(data_set, (extend, extend), 'edge')
    
    weights = np.ones(periods) / periods
    data_set = np.convolve(data_set, weights, mode='valid')

    return data_set

def interpolate_signals(signals: np.ndarray, interpolation_factor: int, fs: float):
    """ Interpolates signals stored in an numpy array. """
    if len(signals) == 0: raise Exception("Signal array must contain signals.")
    if interpolation_factor <= 1: raise Exception("Interpolation factor must be larger than 1.")

    number_of_sig_in_set = signals.shape[0]
    signal_length = signals.shape[1]
    
    timeseries = np.arange(0, signal_length/fs, 1/fs)
    interp_timeseries = np.arange(0, timeseries[-1] + 0.5/(fs * interpolation_factor), 1/(fs * interpolation_factor))
    new_signal_length = len(interp_timeseries)

    interp_signal = np.empty([number_of_sig_in_set, new_signal_length], dtype=float)

    for c in range(number_of_sig_in_set):
        interp_signal[c, :] = interp(interp_timeseries, timeseries, signals[c, :])

    return interp_signal