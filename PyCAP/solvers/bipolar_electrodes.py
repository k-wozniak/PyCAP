import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.linalg import matmul_toeplitz
from typing import Tuple

from PyCAP.solvers.utils.quadratic_solvers import quadratic_solver
from PyCAP.solvers.utils.signal_operations import moving_average

def get_rc(signal, qs_length) -> Tuple[np.ndarray, np.ndarray]:
    r = np.pad(signal, (0, qs_length-1), 'constant')
    c = np.zeros(qs_length)
    c[0] = signal[0]

    return (r, c)

def two_cap(s1, s2, q1, q2, q3, q4) -> np.ndarray:
    """ A two cap implementation as stated in the paper. Uses only two electrodes
        which do not need to be next to each other. Consequently, 4 Qs"""
        
    qs_length = q1.shape[0]

    rc1 = get_rc(s1, qs_length)
    rc2 = get_rc(s2, qs_length)

    Q1 = (q4 - q3)
    Q2 = (q2 - q1)

    C = (matmul_toeplitz(rc1, Q1, check_finite=False) - 
         matmul_toeplitz(rc2, Q2, check_finite=False))

    return quadratic_solver(C)

def mean_two_cap(signals, qs):
    """ Extension of the two cap method where each possible solution to every 
        signal pair is found and the average solution is obtained. More robust
        than the standard two cap method as more electrodes used.
        Possible to extend to ignore outliers. """
    num_signal = len(signals)    
    w = []

    for i in range(0, num_signal):
        c1 = signals[i]
        q1 = qs[i]
        q2 = qs[i+1]

        for j in range(i+1, num_signal):
            c2 = signals[j]
            q3 = qs[j]
            q4 = qs[j+1]

            w.append(two_cap(c1, c2, q1, q2, q3, q4))
    
    w = np.asarray(w)
    w_mean = np.mean(w, axis=0)
    w_mean = moving_average(w_mean, 3)
    w_mean = w_mean/w_mean.sum(axis=0,keepdims=1)
    
    return w_mean

def NCap(signals, qs, initial_values = None, solver_algorithm = quadratic_solver):
    """ Further extension of the two cup method where the C matrix is generated
        based on every avaliable signal. It is also less computation intensive
        than the mean_two_cap. Recommended method to use."""
    qs_length = qs[0].shape[0]
    num_signals = len(signals)

    C = np.zeros((signals[0].shape[0]+qs_length-1, qs[0].shape[1]))
    for i in range(0, num_signals):
        rc1 = get_rc(signals[i], qs_length)
        q1 = qs[i]
        q2 = qs[i+1]

        for j in range(i+1, num_signals):
            rc2 = get_rc(signals[j], qs_length)
            q3 = qs[j]
            q4 = qs[j+1]
            
            C1Q4 = matmul_toeplitz(rc1, q4)
            C1Q3 = matmul_toeplitz(rc1, q3)
            C2Q2 = matmul_toeplitz(rc2, q2)
            C2Q1 = matmul_toeplitz(rc2, q1)

            C = (C + C1Q4 - C1Q3 - C2Q2 + C2Q1)

    return solver_algorithm(C, initial_values)

def NCapPairs(signals, qs, initial_values = None, solver_algorithm = quadratic_solver):
    """ Further extension of the two cup method where the C matrix is generated
        based on every avaliable signal. It is also less computation intensive
        than the mean_two_cap. Recommended method to use."""
    qs_length = qs[0].shape[0]
    num_signals = len(signals)

    # Generate pairs
    middle = round(num_signals/2)
    pairs = ([(0, x) for x in range(middle, num_signals)] + 
        [(num_signals-1, x) for x in range(1, middle+1)])

    C = np.zeros((signals[0].shape[0]+qs_length-1, qs[0].shape[1]))
    for i, j in pairs:
        rc1 = get_rc(signals[i], qs_length)
        rc2 = get_rc(signals[j], qs_length)
        
        q1 = qs[i]
        q2 = qs[i+1]
        q3 = qs[j]
        q4 = qs[j+1]
        
        C1Q4 = matmul_toeplitz(rc1, q4)
        C1Q3 = matmul_toeplitz(rc1, q3)
        C2Q2 = matmul_toeplitz(rc2, q2)
        C2Q1 = matmul_toeplitz(rc2, q1)

        C = (C + C1Q4 - C1Q3 - C2Q2 + C2Q1)


    return solver_algorithm(C, initial_values)

def VSR(d, fs, du, vmin, vstep, vmax) -> np.ndarray:
    """ VSR Delay-and-add recordings - B.W.Metcalfe 2018
            This operates in the time domain and needs to know the electrode
            spacing, the sample rate, and the desired analysis parameters.
            d - Time domain data to beamform
            fs - Sample rate of the data in Hz
            du - The nominal electrode spacing in meters
            vmin - Starting velocity
            vstep - Velocity step
            vmax - Stopping velocity
    """
    d = np.array(d).T
    #d = np.fliplr(d)

    # Number of velocities
    v = np.arange(vmin, vmax, vstep)
    nv = len(v)

    # Get the length of the recordings and the number of channels
    nt, nu = d.shape
        
    # Temporal sampling parameters
    t = np.arange(0, nt) / fs
        
    # Frequency axis
    f = (np.arange(-nt/2, nt/2) / nt) * fs
    
    # Generate element positions
    u = np.arange(0, nu) * du

    urep = np.tile(u, [nt, 1])
        
    dft = fft(d.T).T
    s = 1.0/v # Hmmm

    im = []
    for n in range(0, nv):
        delays = urep * s[n]

        # Delay
        #imn = ifft(ifftshift( fftshift(dft,1) .* exp(-j*2*pi*repmat(f.',1,nu).*delays), 1 ))
        a6 = np.tile(f.T, (nu, 1)).T * delays
        a4 = np.exp(-1j * 2 * np.pi * a6)
        a3 = fftshift(dft, 0)
        a1 = ifftshift(a3 * a4, 0)
        imn = ifft(a1.T).T # Why does it work!!!

        sumation = np.sum(imn.T, axis=0)

        # Sum
        im.append(sumation)
    
    im = abs(np.array(im))
    
    #ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    
    im = im.max(1)

    return im