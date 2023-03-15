from typing import Optional
import numpy as np
from thesis.relaxations.sdp_relaxation_v2 import (
    build_cost_matrix,
    build_homo_constraint,
    build_rotation_constraint_matrices,
    build_measurement_constraint_matrices,
)
from thesis.experiments.utils import StereoLocalizationProblem, StereoLocalizationSolution

class StereoLocalizationCertificate:
    def __init__(self, certified: bool, H: np.array, eig_values: Optional[np.array] = None):
        self.certified = certified 
        self.H = H
        self._eig_values = eig_values

    @property
    def eig_values(self):
        if self._eig_values is None:
            self._eig_values, _ = np.linalg.eig(self.H)
        return self._eig_values

def run_certificate(problem: StereoLocalizationProblem, solution: StereoLocalizationSolution) -> StereoLocalizationCertificate:
    #world: sim.World, y: np.array, solution.T_cw: np.array, r0: np.array, gamma_r: float, W: np.array) -> bool:
    x_1 = solution.T_cw[:3, :].T.reshape((12, 1))
    x_2 = (solution.T_cw @ problem.p_w / np.expand_dims((np.array([0, 0, 1, 0]) @ solution.T_cw @ problem.p_w), -1))[:, [0, 1, 3], :].reshape(-1, 1)
    x_local = np.concatenate((x_1, x_2, np.array([[1]])), axis = 0)

    num_landmarks = problem.p_w.shape[0]
    Ws = np.zeros((num_landmarks, 4, 4))
    for i in range(num_landmarks):
        Ws[i] = problem.W

    # build cost matrix and compare to local solution
    D = 13 + 3*num_landmarks
    Q = build_cost_matrix(D, problem.y, Ws, problem.M, problem.r_0, problem.gamma_r)
    Q = Q / np.mean(np.abs(Q)) # improve numerics 

    # rotation matrix
    As, bs = build_rotation_constraint_matrices(D)

    # homogenization variable
    A, b = build_homo_constraint(D)
    As.append(A)
    bs.append(b)

    # measurements
    A_measure, b_measure = build_measurement_constraint_matrices(D, problem.p_w)
    As += A_measure
    bs += b_measure

    lhs = np.concatenate([A @ x_local for A in As], axis = 1) # \in R^((12 + J*5 + 1), (12 + J*3 + 1))
    rhs = Q @ x_local
    lag_mult = np.linalg.lstsq(lhs, rhs, rcond = None)[0]
    assert np.allclose(lhs @ lag_mult, rhs)
    H = Q - sum([A * lag_mult[i] for i, A in enumerate(As)])
    #np.all(np.linalg.eigvals(H) > 0)
    eig_values, _ = np.linalg.eig(H)
    real_parts = eig_values.real
    imag_parts = eig_values.imag
    
    #print(eig_values)
    certified = (real_parts.min() > -10e-3) and np.allclose(imag_parts, 0)
    return StereoLocalizationCertificate(certified, H, eig_values)

