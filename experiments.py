import numpy as np
from typing import Optional

class StereoLocalizationProblem:
    def __init__(
        self,
        T_wc: np.array,
        p_w: np.array,
        M: np.array,
        W: Optional[np.array] = None,
        y: Optional[np.array] = None,
        gamma_r: float = 0.0,
        r_0: Optional[np.array] = None,
        gamma_C: float = 0.0,
        C_0: Optional[np.array] = None,
    ):
        self.T_wc = T_wc
        self.p_w = p_w
        self.y = y
        self.W = W
        self.M = M
        self.gamma_r = gamma_r
        self.r_0 = r_0
        self.gamma_C = gamma_C
        self.C_0 = C_0

class StereoLocalizationSolution:

    def __init__(self, solved: bool, T_cw: Optional[np.array] = None, cost: Optional[float] = None):
        self.solved = solved
        self.T_cw = T_cw
        self.cost = cost