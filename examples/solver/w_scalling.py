"""
The model and solver is based on the set of equations outlined in the paper attached
to the git repo where wi is obtained using:

wo = A(v)H(m)

m = Mp(v)

v - velocity
A(v) - SFAP amplitude dependence on CV
H(m) - dependence of the weighting coefficients on the number of fibres in CV class i

Sum(m) = M

This means that resulting NCAP returns vector w and it needs to be scaled by A(v) in
oreder to represent the actual nerve fibre velocity distribution

If w = [0.1, 0.1, 0.1, 0.5, 0.2]
Corresponding to the following velocities
cvs = [1, 2, 3, 4, 5]

Based on the most cited amplitude dependence A(v) = v^2

Each w must be divided by corresponding scaling
"""

import numpy as np

w = np.asarray([0.1, 0.1, 0.1, 0.5, 0.2])
cvs = np.asarray([1, 2, 3, 4, 5])

distribution = w / (cvs**2)

# for linear dependency
distribution = w / cvs

# for A(v) = 1
distribution = w / 1 # = w