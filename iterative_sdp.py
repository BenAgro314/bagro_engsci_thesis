from typing import List, Optional, Dict
import os
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from experiments import StereoLocalizationSolution, StereoLocalizationProblem

import numpy as np
import sim
import local_solver
from local_solver import StereoLocalizationProblem, StereoLocalizationSolution, projection_error
from sdp_relaxation import build_rotation_constraint_matrices, build_general_SDP_problem, extract_solution_from_X
from datetime import datetime
import cvxpy as cp

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
    min_update_norm = 1e-10, max_iters = 100,
    return_X: bool = True,
    momentum_param_k: float = None,
    mosek_params: Dict[str, float] = {},
    log: bool = False,
    refine: bool = True,
):
    X_sdp = vec_T(T_init) @ vec_T(T_init).T # remove last row; we already know those values
    T = None
    if momentum_param_k is not None:
        assert momentum_param_k > 0 and momentum_param_k < 1
        T = T_init
    # rotation matrix constraints
    rot_As, bs = build_rotation_constraint_matrices()
    As = []
    for rot_A in rot_As:
        A = np.zeros((13, 13))
        A[:9, :9] = rot_A
        As.append(A)
    # homogenization variable
    A = np.zeros((13, 13))
    A[-1, -1] = 1
    As.append(A)
    bs.append(1)
    X_old = X_sdp
    for i in range(max_iters) :
        Q = _build_Q(problem, _X_sdp_to_X(X_sdp))
        prob, X_var = build_general_SDP_problem(Q, As, bs)
        try:
            prob.solve(solver=cp.MOSEK, mosek_params = mosek_params)#, verbose = True)
            if prob.status != "optimal":
                assert False
        except Exception:
            print("Failed to solve SDP, breaking")
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
            print(f"Small update, breaking on iteration {i + 1}")
            break
        X_old = X_sdp
    if return_X:
        return X_sdp
    else:
        T = extract_solution_from_X(X_sdp)
        if refine:
            return local_solver.stereo_localization_gauss_newton(problem, T, log = False, max_iters = 100)
        else:
            cost = projection_error(problem.y, T, problem.M, problem.p_w, problem.W)
            return StereoLocalizationSolution(True, T, cost)



def main():
    from certificate_noise_test import make_sim_instances

    exp_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, f"outputs/{exp_time}") 
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    var_list = [0.1] #, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9, 10]
    num_problem_instances = 10
    num_landmarks = 20
    num_local_solve_tries = 100 # 40

    cam = sim.Camera(
        f_u = 160, # focal length in horizonal pixels
        f_v = 160, # focal length in vertical pixels
        c_u = 320, # pinhole projection in horizonal pixels
        c_v = 240, # pinhold projection in vertical pixels
        b = 0.25, # baseline (meters)
        R = 0 * np.eye(4), # covarience matrix for image-space noise
        fov = np.array([[-1,1], [-1, 1], [2, 5]])
    )

    p_wc_extent = np.array([[3], [3], [0]])
    instances = make_sim_instances(num_problem_instances, num_landmarks, p_wc_extent, cam)

    world = sim.World(
        cam = cam,
        p_wc_extent = p_wc_extent,
        num_landmarks = num_landmarks,
    )

    metrics = []
    #eps = 1e-7 # try 1e-10
    #mosek_params = {
    #    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps,
    #    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps,
    #    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps,
    #    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800
    #}
    mosek_params = {}

    for var in var_list:
        print(f"Noise Variance: {var}")
        world.cam.R = var * np.eye(4)
        for scene_ind in range(num_problem_instances):
            print(f"Scene ind: {scene_ind}")

            problem = instances[scene_ind]
            problem.y = world.cam.take_picture(problem.T_wc, problem.p_w)
            problem.W = (1/var)*np.eye(4)

            #for _ in tqdm.tqdm(range(num_local_solve_tries)):
            for _ in range(num_local_solve_tries):
                # local solution
                datum = {}
                T_op = sim.generate_random_T(p_wc_extent)
                local_solution = local_solver.stereo_localization_gauss_newton(problem, T_op, log = False, max_iters = 100)
                iter_sdp_soln = iterative_sdp_solution(problem, T_op, max_iters = 1, return_X = False, mosek_params=mosek_params)
                datum["problem"] = problem
                datum["local_solution"] = local_solution
                datum["iterative_sdp_solution"] = iter_sdp_soln
                datum["noise_var"] = var
                datum["scene_ind"] = scene_ind

                metrics.append(datum)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    local_solution_costs = [m["local_solution"].cost for m in metrics]
    iterative_sdp_costs = [m["iterative_sdp_solution"].cost for m in metrics]

    min_cost = min(min(local_solution_costs), min(iterative_sdp_costs))
    max_cost = max(max(local_solution_costs), max(iterative_sdp_costs))
    bins = np.logspace(np.log10(min_cost),np.log10(max_cost), 50)
    axs[0].hist(local_solution_costs, bins=bins)
    axs[0].set_xlabel("Local Solver Solution Cost")
    axs[1].hist(iterative_sdp_costs, bins=bins)
    axs[1].set_xlabel("Iterative SDP Solution Cost")
    fig.subplots_adjust(hspace=0.5)
    plt.savefig("test.png")

    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)



if __name__ == "__main__":
    main()