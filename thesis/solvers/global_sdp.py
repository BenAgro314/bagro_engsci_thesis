from copy import deepcopy
import cvxpy as cp
import numpy as np
from typing import Dict, Optional
from thesis.solvers.local_solver import stereo_localization_gauss_newton, projection_error
from relaxations.sdp_relaxation import build_cost_matrix_v2, build_general_SDP_problem, build_homo_constraint, build_measurement_constraint_matrices_v2, build_rotation_constraint_matrices, extract_solution_from_X
from simulation.sim import generate_random_rot
from thesis.experiments.utils import StereoLocalizationProblem, StereoLocalizationSolution

def global_sdp_solution(
    problem: StereoLocalizationProblem,
    T_init: Optional[np.array] = np.eye(4),
    return_X: bool = True,
    mosek_params: Dict[str, float] = {},
    refine: bool = True,
    max_num_tries: int = 1,
    record_history: bool = False,
):
    success = False
    num_tries = 0
    T_init = deepcopy(T_init)
    T_cw_history = []
    num_landmarks = problem.y.shape[0]
    Ws = np.zeros((num_landmarks, 4, 4))
    for i in range(num_landmarks):
        Ws[i] = problem.W
    while not success and num_tries < max_num_tries:
        success = True

        # build cost matrix and compare to local solution
        Q = build_cost_matrix_v2(num_landmarks, problem.y, Ws, problem.M, problem.r_0, problem.gamma_r)
        Q = Q / np.mean(np.abs(Q)) # improve numerics 
        As = []
        bs = []

        # rotation matrix
        As_rot, bs = build_rotation_constraint_matrices()
        for A_rot in As_rot:
            A = np.zeros((13 + 3*num_landmarks, 13 + 3 *num_landmarks))
            A[:9, :9] = A_rot
            As.append(A)

        # homogenization variable
        A, b = build_homo_constraint(num_landmarks)
        As.append(A)
        bs.append(b)

        # measurements
        A_measure, b_measure = build_measurement_constraint_matrices_v2(problem.p_w)
        As += A_measure
        bs += b_measure

        prob, X = build_general_SDP_problem(Q, As, bs)

        num_tries += 1
        try:
            prob.solve(solver=cp.MOSEK, mosek_params = mosek_params)#, verbose = True)
            if prob.status != "optimal":
                assert False
        except Exception:
            T_init[:3, :3] = generate_random_rot()
            success = False
            continue
        X_sdp = X.value

    if not success:
        print(f"Failed to solve in {max_num_tries} tries")

    if return_X:
        return X_sdp
    else:
        T = extract_solution_from_X(X_sdp)
        T_cw_history.append(T)
        if refine:
            problem = deepcopy(problem)
            problem.T_init = T
            soln = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, record_history = record_history)
            if record_history:
                soln.T_cw_history = T_cw_history + soln.T_cw_history
            return soln
        else:
            cost = projection_error(problem.y, T, problem.M, problem.p_w, problem.W)
            return StereoLocalizationSolution(True, T, cost, T_cw_history)