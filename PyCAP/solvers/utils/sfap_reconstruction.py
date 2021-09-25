"""
The solution algorithm is designed to operate without explicit knowledge of 
SFAPs as the information about them is cancelled out during calculations.
Fortunately, these can be obtained as C = A * Q * w therefore, C*(Qw).T = A
"""

import numpy as np
import warnings
from typing import List

from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def recreate_A_matrix(A: np.ndarray) -> np.ndarray:
    """ The shape of the A matrix is know to unitary matrix as shown below:
    A1  0   0   0   ...     Therefore, by shifting each column up it is possible
    A2  A1  0   0   ...     to align each number. Then by teaking a mean each  
    A3  A2  A1  0   ...     value is found and the matrix is regenerated in 
    A4  A3  A2  0   ...     reverse order to the creation.
    ..  ..  ..  ..  ...     """
    A = np.asmatrix(A)
    cols = A.shape[1]

    # Shift each column by required number of rows
    shifted_A = np.matlib.zeros(A.shape)
    shifted_A[:, 0] = A[:, 0]
    for c in range(1, cols):
        shifted_A[:-c, c] = A[c:, c]
    
    # Take a mean of each row excluding zeros. As the values are floats each 
    # close value to zero is replaced by nan. After taking the mean, nan is 
    # reverted back to 0
    #shifted_A[abs(shifted_A) < 0.000001] = np.nan
    # Warning expected as if all values in the row are NaN it shows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shifted_A = np.nanmean(shifted_A, axis=1)
        
    shifted_A = np.nan_to_num(shifted_A, copy=False, nan=0.0)

    # Recreate A matrix
    new_A = np.matlib.zeros(A.shape)
    new_A[:, 0] = shifted_A[:]
    for c in range(1, cols):
        new_A[c:, c] = shifted_A[:-c]

    return np.asarray(new_A)

def find_sfap_A(signal: np.ndarray, q: np.ndarray, w: np.ndarray, max_sfap_width_samples: int = 500) -> np.ndarray:
    """ Finds the minimum to the problem of Ax = B, where 
    B = signal
    A = q*w
    x = A matrix from the signal equation
    signal = A*Q*w

    For simple bipolar recording Q = Q2 - Q1
    """
    signal = np.array(signal)
    q = np.array(q)
    w = np.array(w)

    # Check the signals shape should be Kx1
    signal = signal.T if signal.shape[0] == 1 else signal
    K = signal.shape[0]

    x, P, N = generate_set_of_equations(K, q, w, max_sfap_width_samples)

    A = np.linalg.lstsq(x, signal, rcond=None)[0]
    A = np.expand_dims(A, axis=1)

    if (A.shape[0] != P) and (K-N > 0):
        A = np.concatenate([A, np.zeros((K-A.shape[0],1))])
    
    A = toeplitz(A, np.zeros((P, 1)))

    return A

def find_sfap_A_from_a_set(signals: np.ndarray, qs: np.ndarray, w: np.ndarray, max_sfap_width_samples: int = 300) -> np.ndarray:

    x = np.array([])
    K = signals[0].shape[0]
    N = 0
    P = 0

    for i in range(len(signals)):
        x_temp, P, N = generate_set_of_equations(K, qs[i], w, max_sfap_width_samples)

        if x.size == 0:
            x = x_temp
        else:
            x = np.concatenate((x, x_temp), axis=0)

    B = np.concatenate(signals)    

    A = np.linalg.lstsq(x, B, rcond=None)[0]
    A = np.expand_dims(A, axis=1)

    if (A.shape[0] != P) and (K-N > 0):
        A = np.concatenate([A, np.zeros((K-A.shape[0],1))])
    
    A = toeplitz(A, np.zeros((P, 1)))

    return A


def generate_set_of_equations(K: int, q: np.ndarray, w: np.ndarray, max_sfap_width_samples: int = 500):
    q = np.array(q)
    w = np.array(w)

    # Check w shape should be Nx1
    # Won't work if there is just one N but that defeats the point
    w = w.T if w.shape[0] == 1 else w
    N = w.shape[0]

    # Check a qs shape should be PxN (Assumes all qs are the same)
    P = q.shape[0]
    Nq = q.shape[1]

    if Nq != N and P == N:
        q = q.T 
        Nq = q.shape[1]

    elif Nq != N and P != N:
        raise Exception("Incorrectly shaped matrices.")

    qw = q@w

    if (K != P) and (K-N > 0):
        qw = np.concatenate([qw, np.zeros((K-N,1))])
    
    return (toeplitz(qw, np.zeros((max_sfap_width_samples,1))), P, N)