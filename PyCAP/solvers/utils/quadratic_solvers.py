import numpy as np
from scipy import sparse
import osqp
import math


def quadratic_solver(C: np.ndarray, initial_values = None):
    """ Uses osqp solver to find the optimal solution for the problem given in 
        matrix C. Additionally, restricts the problem solution that 
        0 <= x <= 1
        Sum(x) = 1
    """
    P = np.matmul(C.T, C)
    length = int(P.shape[0]) # Used to generate boundaries

    # Q is zeros as there is no linear part of the problem
    q = np.zeros((length, 1))

    # Constrains
    # Each single X must be between 0 and 1
    # Achieved using a identity matrix where each value has min of 0 and max 1
    # Additional constrain that all X added must equal to 1
    # This is achieved by adding one more row where min = 1, max = 1, A = 1
    # So that 1 <= Ax <= 1 so Sum(x) = 1
    constrain = 1

    A = np.append(np.identity(length), np.ones((1, length)), axis=0)
    lower_boundary = np.zeros((length + constrain, 1))
    upper_boundary = np.ones((length + constrain, 1))

    lower_boundary[-constrain] = 1
    
    # Convert both to sparse as required
    P = sparse.csc_matrix(P) 
    A = sparse.csc_matrix(A)

    # Generate model
    model = osqp.OSQP()
    model.setup(P=P, q=q, A=A, l=lower_boundary, u=upper_boundary, verbose=False)
    
    # Possible to start with initial values, like an gaussian distribution
    if initial_values is not None:
        model.warm_start(x=initial_values)
    
    # Solve and return the model
    results = model.solve()
    return results.x


def cumminssolver_helper(C: np.ndarray, initial_values = None, initialise=False):
    if initialise:
        pass
    return cumminssolver(C, initial_values=initial_values)


def cumminssolver(C: np.ndarray, threshold: float = 0.01, stagnation: int = 10, initial_values: np.ndarray = None):
    """ TODO Implement solver presented in the paper """
    
    if stagnation < 1:
        raise ValueError

    # Convert C to matrix to automatically use matrix multiplication
    # Mafalda changes: kept this as ndarray since np.matrix is being deprecated
    # so, changed a lot of lines to np.dot() to perform multiplication correctly
    C = 2*(np.dot(C.T, C))

    w_length = C.shape[1]
    # Possible to start with initial values, like an gaussian distribution
    if initial_values is None:
        # Initialise to all having identical probabilities
        w = np.ones((w_length, 1)) / w_length # Possibly wrong dimension
    else:
        # Check if initial_values has correct dimensions
        if initial_values.shape[0] == w_length and initial_values.shape[1] == 1:
            w = initial_values
        elif initial_values.shape[0] == 1 and initial_values.shape[1] == w_length:
            w = initial_values.T
        else:
            raise Exception('Incorrect Initial Values passed')

    stagnation_count = 0
    while stagnation_count <= stagnation:
        # Calculate current estimate of lambda
        k = np.ones((w_length, 1))
        l = np.dot(np.dot(w.T, C), w) / np.dot(w.T, k)**2

        # Find true gradient
        gradient = np.dot(C, w) - np.multiply(l, k)

        if np.all(gradient < threshold):
            return np.squeeze(np.asarray(w))

        # Determine modified gradient (with non-negativity constraint)
        for i in range(0, gradient.shape[0]):
            if gradient[i] > 0:
                gradient[i] = (1 - math.exp(-w_length*w[i])) * gradient[i]

        alpha = np.dot(gradient.T, gradient) / np.dot(np.dot(gradient.T, C), gradient)

        # Calculate the new w vector and f(w)
        w_new = w - np.multiply(alpha, gradient)

        # Evaluate objective function to see if convergence has been reached
        objective_function = lambda w, C, k : (0.5 * np.dot(np.dot(w.T, C), w) + np.dot(l, (1-np.dot(w.T, k))))
        old_solution = objective_function(w, C, k)
        new_solution = objective_function(w_new, C, k)
        print("Current minimum: ", new_solution)
        print("-----")

        # Check if solution is constant for at least 10 consecutive iterations
        percentageChange = abs((new_solution - old_solution)/old_solution)*100

        if percentageChange <= threshold:
            stagnation_count = stagnation_count + 1
        else:
            stagnation_count = 0

        w = w_new

    return np.squeeze(np.asarray(w))