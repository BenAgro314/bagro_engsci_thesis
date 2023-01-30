from copy import deepcopy
from typing import Dict, Optional, Tuple, List

import cvxpy as cp
import numpy as np
from thesis.experiments.utils import StereoLocalizationProblem, StereoLocalizationSolution
from scipy.linalg import fractional_matrix_power
from thesis.solvers.local_solver import projection_error, stereo_localization_gauss_newton
from thesis.relaxations.sdp_relaxation import (
    build_general_SDP_problem, build_rotation_constraint_matrices, Ei,
    extract_solution_from_X)

def _build_Q(problem: StereoLocalizationProblem) -> np.array:
    assert problem.y is not None and problem.T_wc is not None and problem.p_w is not None
    assert problem.W is not None, problem.M is not None
    Q = 0
    a = np.array([[0], [0], [1], [0]])
    N = problem.p_w.shape[0]
    if len(problem.W.shape) == 2:
        W = np.stack([problem.W] * N, axis = 0) # (N, 4, 4)
    for k in range(N):
        p_k = problem.p_w[k]
        y_k = problem.y[k]
        Q += (np.kron(p_k.T, np.eye(4)).T @ (y_k @ a.T - problem.M).T @ W[k] @ (y_k @ a.T - problem.M) @ (np.kron(p_k.T, np.eye(4))))
    # remove uneeded rows and colums (4, 8, 12)
    for a in [0, 1]:
        Q = np.delete(Q, [3, 7, 11], axis = a)
    Q = np.concatenate((Q, np.zeros_like(Q[:, -1:])), axis = 1)
    Q = np.concatenate((Q, np.zeros_like(Q[-1:, :])), axis = 0) # add zero 
    return Q

def build_rotation_constraints(dim) -> Tuple[List[np.array], List[np.array]]:
    """Builds the 6 9x9 rotation matrix constraint matrices

    Returns:
        Tuple[List[np.array], List[np.array]]: First element in the tuple is a list of rotation matrix
        constraint matrices. Second element in the tuple is a list of the rhs of those constraint equations
        (b in Ax = b)
    """
    As = []
    bs = []
    for i in range(0, 3):
        for j in range(i, 3):
            A = np.zeros((dim, dim))
            A = Ei(j, dim = dim).T @ Ei(i, dim = dim)
            A[-1, -1] = -int(i == j)
            As.append(0.5 * (A + A.T))
            bs.append(0)

    return As, bs

def unknown_scale_sdp(
    problem: StereoLocalizationProblem,
    return_X: bool = True,
    mosek_params: Dict[str, float] = {},
    refine: bool = True,
    record_history: bool = False,
    log: bool = False,
):
    success = False
    T_cw_history = []
    num_landmarks = problem.y.shape[0]
    Ws = np.zeros((num_landmarks, 4, 4))
    for i in range(num_landmarks):
        Ws[i] = problem.W

    success = True
    Q = _build_Q(problem)
    Q = Q / np.mean(np.abs(Q))
    As, bs = build_rotation_constraints(dim = 14)


    prob, X = build_general_SDP_problem(Q, As, bs)

    try:
        prob.solve(solver=cp.MOSEK, mosek_params = mosek_params)#, verbose = True)
        if prob.status != "optimal":
            assert False
        if log:
            print("The optimal value from the SDP is", prob.value)
    except Exception:
        success = False
    X_sdp = X.value

    if not success:
        print(f"Global SDP failed to solve tries")

    #if return_X:
    return X_sdp
    #else:
    #    X_sdp = X_sdp / X_sdp
    #    T = extract_solution_from_X(X_sdp)
    #    T_cw_history.append(T)
    #    if refine:
    #        problem = deepcopy(problem)
    #        problem.T_init = T
    #        soln = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, record_history = record_history)
    #        if record_history:
    #            soln.T_cw_history = T_cw_history + soln.T_cw_history
    #        return soln
    #    else:
    #        cost = projection_error(problem.y, T, problem.M, problem.p_w, problem.W)
    #        return StereoLocalizationSolution(True, T, cost, T_cw_history)