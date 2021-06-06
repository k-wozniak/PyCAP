import numpy as np
from enum import Enum

class DistributionType(Enum):
    Uniform = 1
    Step = 2
    SudoRandom = 3

def create_simple_distribution(dmin: float, dmax: float, step: float, dtype: DistributionType):
    cv_dis = np.arange(dmin, (dmax + step), step)
    
    if isinstance(DistributionType.Uniform, dtype):
        fibres = np.ones(cv_dis.shape[0])
    elif isinstance(DistributionType.Step, dtype):
        fibres = cv_dis.copy()
    elif isinstance(DistributionType.SudoRandom, dtype):
        fibres = np.array([1, 1, 1, 1, 4, 4, 5, 4, 10, 10, 1, 2, 2, 1, 5, 6, 4, 4, 2, 2, 3])
    else:
        raise NotImplementedError("No such functionality.")
    
    return np.c_[cv_dis, fibres]