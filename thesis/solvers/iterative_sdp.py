from copy import deepcopy
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
from thesis.experiments.utils import StereoLocalizationProblem, StereoLocalizationSolution
from scipy.linalg import fractional_matrix_power
from thesis.solvers.local_solver import projection_error, stereo_localization_gauss_newton
from thesis.relaxations.sdp_relaxation_v2 import (
    build_general_SDP_problem, build_rotation_constraint_matrices,
    extract_solution_from_X)
from thesis.simulation.sim import generate_random_rot, generate_random_T

RECORD_HISTORY=True

def vec(A: np.array) -> np.array:
    assert len(A.shape) == 2
    res = A.T.reshape(-1, 1)
    assert np.all(A[:, 0] == res[:A.shape[0], 0])
    return res

def vec_T(T: np.array) -> np.array:
    v = vec(T) # remove last row
    v = np.delete(v, [3, 7, 11], axis = 0)
    return v

def _X_sdp_to_X(X_sdp: np.array) -> np.array:
    # input is of shape (13, 13), output is of shape (16, 16) (adding back the zeros rows / columns)
    X = X_sdp.copy()
    for a in [0, 1]:
        X = np.insert(X, [3, 7, 11], 0, axis = a)
    return X

def z_k_sq_from_X(p: np.array, X: np.array) -> float:
    assert p.shape == (4, 1)
    assert p[-1, 0] == 1
    a = np.array([[0], [0], [1], [0]])
    return np.trace(np.kron(p.T, np.eye(4)).T @ a @ a.T @ np.kron(p.T, np.eye(4)) @ X)

def _build_Q(problem: StereoLocalizationProblem, X_prev: np.array) -> np.array:
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
        z_k_sq = z_k_sq_from_X(p_k, X_prev)
        Q += (1/z_k_sq) * (np.kron(p_k.T, np.eye(4)).T @ (y_k @ a.T - problem.M).T @ W[k] @ (y_k @ a.T - problem.M) @ (np.kron(p_k.T, np.eye(4))))
    # remove uneeded rows and colums (4, 8, 12)
    for a in [0, 1]:
        Q = np.delete(Q, [3, 7, 11], axis = a)
    return Q

def iterative_sdp_solution(
    problem: StereoLocalizationProblem,
    T_init: Optional[np.array] = np.eye(4),
    min_update_norm = 1e-10,
    max_iters = 100,
    return_X: bool = True,
    momentum_param_k: float = None,
    mosek_params: Dict[str, float] = {},
    log: bool = False,
    refine: bool = True,
    max_num_tries: int = 1,
    record_history: bool = False,
):
    success = False
    num_tries = 0
    T_init = deepcopy(T_init)
    primal_cost = None
    while not success and num_tries < max_num_tries:
        if log:
            print(f"Try number {num_tries + 1}")
        success = True
        X_sdp = vec_T(T_init) @ vec_T(T_init).T # remove last row; we already know those values
        T = None
        if momentum_param_k is not None:
            assert momentum_param_k > 0 and momentum_param_k < 1
            T = T_init
        # rotation matrix constraints
        D = 13
        As, bs = build_rotation_constraint_matrices(D)
        # homogenization variable
        A = np.zeros((D, D))
        A[-1, -1] = 1
        As.append(A)
        bs.append(1)
        X_old = X_sdp
        num_tries += 1
        T_cw_history = [] if record_history else None
        for i in range(max_iters):
            if log:
                print(f"Iteration {i + 1}")
            if record_history:
                T_cw_history.append(extract_solution_from_X(X_sdp))
            Q = _build_Q(problem, _X_sdp_to_X(X_sdp))
            prob, X_var = build_general_SDP_problem(Q, As, bs)
            try:
                prob.solve(mosek_params = mosek_params, verbose = False)
                primal_cost = prob.value
                if prob.status != "optimal":
                    assert False
            except Exception as e:
                if log:
                    print(e)
                T_init[:3, :3] = generate_random_rot()
                success = False
                break
            X_sdp = X_var.value
            if momentum_param_k is not None:
                T_sdp = extract_solution_from_X(X_sdp)
                T = fractional_matrix_power(np.linalg.inv(T) @ T_sdp, momentum_param_k).real @ T
                X_sdp = vec_T(T) @ vec_T(T).T
            assert X_sdp is not None, f"{prob.status}, {X_sdp}"
            assert X_old is not None, f"{prob.status}, {X_old}"
            norm = np.linalg.norm(X_sdp - X_old, ord = "fro")
            if log:
                print(f"Update norm: {norm}")
            if norm < min_update_norm:
                if log:
                    print(f"Small update, breaking on iteration {i + 1}")
                break
            X_old = X_sdp

    if not success:
        print(f"Iterative SDP failed to solve in {max_num_tries} tries")

    if return_X:
        return X_sdp
    else:
        T = extract_solution_from_X(X_sdp)
        if refine:
            problem = deepcopy(problem)
            problem.T_init = T
            soln = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, record_history = record_history)
            if record_history:
                soln.T_cw_history = T_cw_history + soln.T_cw_history
            soln.primal_cost = primal_cost
            return soln
        else:
            if record_history:
                T_cw_history.append(T)
            cost = projection_error(problem.y, T, problem.M, problem.p_w, problem.W)
            return StereoLocalizationSolution(True, T, cost, T_cw_history, primal_cost = primal_cost)