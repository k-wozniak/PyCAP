from scipy.linalg import toeplitz
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