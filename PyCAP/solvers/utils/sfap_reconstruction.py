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

def find_matrix_A(signal: np.ndarray, q: np.ndarray, w: np.ndarray):
    """ Finds the minimum to the problem of xA = B, where 
    B = signal
    A = q*w
    x = A matrix from the signal equation
    signal = A*Q*w

    For simple bipolar recording Q = Q2 - Q1
    """
    signal = np.asmatrix(signal)
    q = np.asmatrix(q)
    w = np.asmatrix(w)

    # Check the signals shape should be Kx1
    signal = signal.T if signal.shape[0] == 1 else signal

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

    qw = np.matmul(q, w)
    out = np.linalg.lstsq(qw.T, signal.T, rcond=None)[0]
    
    return out.T

def find_matrix_A_from_set(signals: List[np.ndarray], qs: List[np.ndarray], w: np.ndarray):
    """ Finds matrix A for each avaliable signal.
        Then, ignoring zeros, finds the mean value for each cell producing
        mean matrix A.
    """
    # Find matrix A for each signal
    sfaps = []
    for i in range(len(signals)):
        sfaps.append(find_matrix_A(signals[i], qs[i], w))

    sfaps = [ np.asarray(x) for x in sfaps ] # Change from matrix to an array
    sfaps = np.dstack(sfaps) # Combine into 3D array
    # As the array is float, all near zero values are replaced with
    # np.nan to be ignored during mean operation
    sfaps[abs(sfaps) < 0.000001] = np.nan 

    # Warning expected as if all values in the row are NaN it shows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sfaps = np.nanmean(sfaps, axis=2)
    
    sfaps = np.nan_to_num(sfaps, copy=False, nan=0.0) # Convert Nan back to 0s

    return sfaps

def recreate_A_matrix(A: np.ndarray):
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

'''
Functions below could be potentially used. Currently untested.

def deltaU(A):
    """ Calculate the operator norm distance \Vert\hat{A} - \hat{U}\Vert between
        an arbitrary matrix $\hat{A}$ and the closest unitary matrix $\hat{U}$
    """
    __, S, __ = scipy.linalg.svd(A)
    d = 0.0
    for s in S:
        if abs(s - 1.0) > d:
            d = abs(s-1.0)
    return d

def closest_unitary(A):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.

        Return U as a numpy matrix.
    """
    V, __, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U
'''

def find_sfap_A(signal: np.ndarray, q: np.ndarray, w: np.ndarray):
    """ Finds the minimum to the problem of xA = B, where 
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

    t = toeplitz(np.ones((P,1)), np.zeros((P,1)))

    x = t * qw.T # The order is very important

    for r in range(1, x.shape[0]):
        x[r, :r+1] = np.flip(x[r, :r+1])

    A = np.linalg.lstsq(x, signal.T, rcond=None)[0]
    A = toeplitz(A, np.zeros((len(A), 1)))
    #A = np.linalg.solve(x, signal)

    
    # Generate teoplitz matrix with ones and zeros only
    # Needed to find linear equations

    # Dot multiply the teoplitz matrix with the A matrix

    # Reverse order of each row of the matrix by taking all 
    # Non zero values and placing them in in a flipped version

    # Use linear solver to find the values

    # Recreate A matrix

    return A

def find_sfap_A_set(signals: np.ndarray, qs: np.ndarray, w: np.ndarray):
    S = np.sum(signals, axis=0)

    P = qs[0].shape[0]
    T = np.zeros(P)
    for x in range(len(signals)):
        t = toeplitz(np.ones((P,1)), np.zeros((P,1)))
        t = t * ((qs[x+1]-qs[x])@w)

        for r in range(1, t.shape[0]):
            t[r, :r+1] = np.flip(t[r, :r+1])

        T = T + t
    
    A = np.linalg.lstsq(T, S)[0]
    A = toeplitz(A, np.zeros((len(A), 1)))

    return A
