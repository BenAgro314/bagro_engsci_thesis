import scipy.io
import numpy as np
from pylgmath.so3.operations import hat
import os
import pickle
from experiments import StereoLocalizationProblem
import local_solver
from iterative_sdp import iterative_sdp_solution
from sim import Camera
import plotting
import sim
from typing import List
from datetime import datetime
from copy import deepcopy

RECORD_HISTORY = False

def read_dataset(path: str):
    dataset3 = scipy.io.loadmat(path)

    # F_i: inertial frame
    # F_v: vehicle frame
    # F_c: camera frame

    theta_vk_i = dataset3["theta_vk_i"] # a 3xK matrix
    r_i_vk_i = dataset3["r_i_vk_i"] # a 3xK matrix where the kth column is the groundtruth position of the camera at timestep k

    y_k_j = dataset3["y_k_j"] # 4 x K x 20 array of observations. All components of y_k_j(:, k, j) will be -1 if the observation is invalid
    y_var = dataset3["y_var"] # 4 x 1 matrix of computed variances based on ground truth stereo measurements
    rho_i_pj_i = dataset3["rho_i_pj_i"] # a 3x20 matrix where the jth column is the poisition of feature j

    # camera to vehicle
    C_c_v = dataset3["C_c_v"] # 3 x 3 matrix giving rotation from vehicle frame to camera frame
    rho_v_c_v = dataset3["rho_v_c_v"]

    # intrinsics
    fu = dataset3["fu"]
    fv = dataset3["fv"]
    cu = dataset3["cu"]
    cv = dataset3["cv"]
    b = dataset3["b"]

    cam = Camera(fu, fv, cu, cv, b, 0 * np.eye(4), None) # is this noise correct?
    M = np.array(cam.M(), dtype=np.float64)

    T_c_v = np.eye(4)
    T_c_v[:3, :3] = C_c_v
    T_c_v[:3, -1:] = -C_c_v @ rho_v_c_v

    problems = []

    p_max = -np.inf * np.ones((3, 1))
    W = np.array(np.linalg.inv(np.diag(y_var.reshape(-1))), dtype=np.float64)
    #W = np.ones((4, 4), dtype=np.float64)

    for k, psi in enumerate(theta_vk_i.T):

        p_w = rho_i_pj_i.T[:, :, None]
        p_w = np.concatenate((p_w, np.ones_like(p_w[:, 0:1, :])), axis = 1)
        ys_with_invalid = y_k_j[:, k, :].T[:, :, None]
        mask = ~((ys_with_invalid.squeeze(-1) == -1).all(axis = 1))
        p_w = np.array(p_w[mask] , dtype=np.float64)

        if p_w.shape[0] < 3: # reject problems with less than 3 landmark points in the image
            continue

        y = np.array(ys_with_invalid[mask], dtype=np.float64)

        psi = psi.reshape(3, 1)
        psi_mag = np.linalg.norm(psi)
        C_vk_i = np.cos(psi_mag) * np.eye(3) + ( 1 - np.cos(psi_mag) ) * (psi / psi_mag) @ (psi / psi_mag).T - np.sin(psi_mag) * hat(psi / psi_mag)
        T_vk_i = np.eye(4)
        T_vk_i[:3, :3] = C_vk_i
        T_vk_i[:3, -1] = - C_vk_i @ r_i_vk_i[:, k]
        T_ck_i = T_c_v @ T_vk_i

        T_w_ck = np.array(np.linalg.inv(T_ck_i), dtype = np.float64)
        problems.append(
            StereoLocalizationProblem(
                T_wc = T_w_ck,
                p_w = p_w,
                M = M,
                y = y,
                W = W,
            )
        )
        p_max = np.maximum(p_max, T_w_ck[:3, -1:])

    p_wc_extent = p_max

    return problems, p_wc_extent, np.diag(y_var)

def metrics_fcn(problem, num_tries = 100):
    mosek_params = {}
    datum = {}
    local_solution = local_solver.stereo_localization_gauss_newton(problem, log = False, max_iters = 100, num_tries = num_tries, record_history=RECORD_HISTORY)
    iter_sdp_soln = iterative_sdp_solution(problem, problem.T_init, max_iters = 1, min_update_norm = 1e-10, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY)
    datum["local_solution"] = local_solution
    datum["iterative_sdp_solution"] = iter_sdp_soln

    return datum

def run_experiment(problems: List[StereoLocalizationProblem], num_local_solve_tries: int, p_wc_extent: np.array):

    exp_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, f"outputs/{exp_time}")
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    metrics = []

    for ind, problem in enumerate(problems):
        print(f"Problem {ind+1}/{len(problems)}")
        for _ in range(num_local_solve_tries):
            T_op = sim.generate_random_T(p_wc_extent)
            problem.T_init = T_op
            metrics.append(metrics_fcn(problem))
            metrics[-1]["problem"] = deepcopy(problem)
            metrics[-1]['scene_ind'] = ind
            metrics[-1]['noise_var'] = 'natural'

    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    return metrics, exp_dir


def main():
    num_local_solve_tries = 20

    problems, p_wc_extent, y_var = read_dataset('dataset3/dataset3.mat')
    metrics, exp_dir = run_experiment(problems, num_local_solve_tries, p_wc_extent)
    plotting.plot_percent_succ_vs_noise(metrics, os.path.join(exp_dir, "local_vs_iterative_bar.png"))

if __name__ == "__main__":
    main()