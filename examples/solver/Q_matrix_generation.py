"""
F = AQ where Q is simply a PxN matrix with shaping functions for each velocity.
In case where all velocity classes have identical SFAP shapes, all elements of matrix
become zero except for a ones located at the proper position to yield the delay di
in the function fi(tk - di).
The specification of the function may be extended to involve SFAP varaitions but the
basic concepts are shown in this example.
"""

import PyCAP.solvers.utils.qs_generation as qs
import numpy as np

cv_min, cv_max, cv_step = (5, 10, 1)
cv_dis_range = np.arange(cv_min, cv_max + cv_step, cv_step)

Q = qs.generate_q(15, 0.05, cv_dis_range, 1000)

print(Q)