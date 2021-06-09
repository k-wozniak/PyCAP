import numpy as np

from PyCAP.solvers.utils.quadratic_solvers import quadratic_solver
from PyCAP.solvers.utils.signal_operations import extend_signals_toeplitz, moving_average

def two_cap(c1, c2, q1, q2, q3, q4):
    """ A two cap implementation as stated in the paper. Uses only two electrodes
        which do not need to be next to each other. Consequently, 4 Qs"""
    conv_signals = extend_signals_toeplitz([c1, c2], q1.shape[0])

    Q1 = (q4 - q3)
    Q2 = (q2 - q1)
    
    C = np.matmul(conv_signals[0], Q1) - np.matmul(conv_signals[1], Q2)

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

    conv_signals = extend_signals_toeplitz(signals, qs_length)
    
    C = np.zeros((conv_signals[0].shape[0], qs[0].shape[1]))
    for i in range(0, num_signals):
        c1 = conv_signals[i]
        q1 = qs[i]
        q2 = qs[i+1]

        for j in range(i+1, num_signals):
            c2 = conv_signals[j]
            q3 = qs[j]
            q4 = qs[j+1]
            
            C1Q4 = np.matmul(c1, q4)
            C1Q3 = np.matmul(c1, q3)
            C2Q2 = np.matmul(c2, q2)
            C2Q1 = np.matmul(c2, q1)

            C = (C + C1Q4 - C1Q3 - C2Q2 + C2Q1)

    return solver_algorithm(C, initial_values)

def NCapPairs(signals, qs, initial_values = None, solver_algorithm = quadratic_solver):
    """ Further extension of the two cup method where the C matrix is generated
        based on every avaliable signal. It is also less computation intensive
        than the mean_two_cap. Recommended method to use."""
    qs_length = qs[0].shape[0]
    num_signals = len(signals)

    conv_signals = extend_signals_toeplitz(signals, qs_length)
    
    C = np.zeros((conv_signals[0].shape[0], qs[0].shape[1]))
    for i in range(0, num_signals-1, 2):
        c1 = conv_signals[i]
        q1 = qs[i]
        q2 = qs[i+1]

        j = i+1
        
        c2 = conv_signals[j]
        q3 = qs[j]
        q4 = qs[j+1]
        
        C1Q4 = np.matmul(c1, q4)
        C1Q3 = np.matmul(c1, q3)
        C2Q2 = np.matmul(c2, q2)
        C2Q1 = np.matmul(c2, q1)

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
    d = np.array(d)
    d = np.fliplr(d)

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
        
    dft = np.fft.fft(d.T).T
    s = 1.0/v # Hmmm

    im = []
    for n in range(0, nv):
        delays = urep * s[n]

        # Delay
        #imn = ifft(ifftshift( fftshift(dft,1) .* exp(-j*2*pi*repmat(f.',1,nu).*delays), 1 ))
        a6 = np.tile(f.T, (nu, 1))
        a5 = a6.T * delays
        a4 = np.exp(-1j * 2 * np.pi * a5)
        a3 = np.fft.fftshift(dft, 0)
        a2 = a3 * a4
        a1 = np.fft.ifftshift(a2, 0)
        imn = np.fft.ifft(a1.T).T # Why does it work!!!!!!!!!!!!!!!!!!!!!!!!!!

        sumation = np.sum(imn.T, axis=0)

        # Sum
        im.append(sumation)
    
    im = abs(np.array(im))
    
    #ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
    
    im = im.max(1)

    return im