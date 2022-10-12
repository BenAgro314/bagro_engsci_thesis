import os 
import pickle
from datetime import datetime
from turtle import color
from typing import Dict, Optional, Tuple, List
import tqdm
from itertools import product

import cvxpy as cp
import numpy as np
import pylgmath
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from pylgmath.so3.operations import vec2rot
import plotting
import sim
import local_solver
from sdp_relaxation import (
    build_general_SDP_problem,
    block_diagonal,
    build_cost_matrix,
    build_rotation_constraint_matrices,
    build_measurement_constraint_matrices,
)
import mosek

def run_certificate(world: sim.World, y: np.array, T_op: np.array, r0: np.array, gamma_r: float, W: np.array) -> bool:
    x_1 = T_op[:3, :].T.reshape((12, 1))
    x_2 = (T_op @ world.p_w / np.expand_dims((np.array([0, 0, 1, 0]) @ T_op @ world.p_w), -1))[:, [0, 1, 3], :].reshape(-1, 1)
    x_local = np.concatenate((x_1, x_2, np.array([[1]])), axis = 0)

    Ws = np.zeros((world.num_landmarks, 4, 4))
    for i in range(world.num_landmarks):
        Ws[i] = W

    # build cost matrix and compare to local solution
    n = 13 + 3 * world.num_landmarks
    Q = build_cost_matrix(world.num_landmarks, y, Ws, world.cam.M(), r0, gamma_r)
    As = []
    bs = []

    # rotation matrix
    rot_matrix_As, bs = build_rotation_constraint_matrices()
    for rot_matrix_A in rot_matrix_As:
        A = np.zeros((n, n))
        A[:9, :9] = rot_matrix_A
        As.append(A)

    # homogenization variable
    A = np.zeros((n, n))
    A[-1, -1] = 1
    As.append(A)
    bs.append(1)

    # measurements
    A_measure, b_measure = build_measurement_constraint_matrices(world.num_landmarks, world.p_w)
    As += A_measure
    bs += b_measure

    lhs = np.concatenate([A @ x_local for A in As], axis = 1) # \in R^((12 + J*5 + 1), (12 + J*3 + 1))
    rhs = Q @ x_local
    lag_mult = np.linalg.lstsq(lhs, rhs, rcond = None)[0]
    lag_mult.shape
    H = Q - sum([A * lag_mult[i] for i, A in enumerate(As)])
    #np.all(np.linalg.eigvals(H) > 0)
    eig_values, _ = np.linalg.eig(H)
    real_parts = eig_values.real
    imag_parts = eig_values.imag
    
    #print(eig_values)
    certificate = (real_parts.min() > -10e-3) and np.allclose(imag_parts, 0)
    return certificate, H, eig_values


def make_sim_instances(num_instances: int, num_landmarks: int, p_wc_extent: np.array, cam: sim.Camera) -> List[Tuple[np.array, np.array]]:
    instances = []
    for _ in range(num_instances):
        world = sim.World(
            cam = cam,
            p_wc_extent = p_wc_extent,
            num_landmarks = num_landmarks,
        )
        world.clear_sim_instance()
        world.make_random_sim_instance()
        instances.append((world.T_wc, world.p_w))
    return instances


def main():
    """Experiment Outline:
        1. Iterate over noise levels and number of points 
    """

    exp_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, f"outputs/{exp_time}") 
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    var_list = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
    num_instances = 1 #10# 50
    num_landmarks = 10
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
    instances = make_sim_instances(num_instances, num_landmarks, p_wc_extent, cam)
    num_local_solves = 100 #50
    r0 = np.zeros((3, 1))
    gamma_r = 0
    W = np.eye(4)
    world = sim.World(
        cam = cam,
        p_wc_extent = p_wc_extent,
        num_landmarks = num_landmarks,
    )
    num_certified_per_var = {}
    local_solutions = {}
    best_local_solutions = {}
    saved_solutions = {} 
    for var in var_list:
        local_solutions[var] = {}
        num_certified_per_var[var] = 0
        best_local_solutions[var] = {}
        saved_solutions[var] = {}
        world.cam.R = var * np.eye(4)
        print(f"Noise Variance: {var}")
        for scene_ind in range(num_instances):
            saved_solutions[var][scene_ind] = {}
            world.T_wc = instances[scene_ind][0]
            world.p_w = instances[scene_ind][1]
            print(f"Scene ind: {scene_ind}")
            y = world.cam.take_picture(world.T_wc, world.p_w)
            best_solution_ind = None
            min_cost = float('inf')
            local_solutions[var][scene_ind] = []

            for i in tqdm.tqdm(range(num_local_solves)):
                # local solution
                T_op = sim.generate_random_T(p_wc_extent)
                T_op, local_minima = local_solver.stereo_localization_gauss_newton(
                    T_op, y, world.p_w, W, world.cam.M(), r_0 = r0, gamma_r = gamma_r, log = False
                )
                if local_minima < min_cost:
                    best_solution_ind = i
                    min_cost = local_minima
                certificate, H, eig_values = run_certificate(world, y, T_op, r0, gamma_r, W)
                local_solutions[var][scene_ind].append((T_op, local_minima, certificate, eig_values, H))

            best_local_solutions[var][scene_ind] = best_solution_ind

            print(f"Certificate: {certificate}")
            if certificate:
                num_certified_per_var[var] += local_solutions[var][scene_ind][best_solution_ind][2]

            saved_solutions[var][scene_ind]["best_local_solution"] = local_solutions[var][scene_ind][best_solution_ind][0]
            saved_solutions[var][scene_ind]["world"] = world
            saved_solutions[var][scene_ind]["y"] = y
            saved_solutions[var][scene_ind]["local_minima"] = local_solutions[var][scene_ind][best_solution_ind][1]
            saved_solutions[var][scene_ind]["certificate"] = local_solutions[var][scene_ind][best_solution_ind][2]
            saved_solutions[var][scene_ind]["H"] = local_solutions[var][scene_ind][best_solution_ind][4]

    with open(os.path.join(exp_dir, f"saved_solutions.pkl"), "wb") as f:
        pickle.dump(saved_solutions, f, protocol=pickle.HIGHEST_PROTOCOL)

    for var in local_solutions:
        for scene_ind in local_solutions[var]:
            best_solution_ind = best_local_solutions[var][scene_ind]
            best_minima = local_solutions[var][scene_ind][best_solution_ind][1]
            colors = ["b" if np.isclose(v[1], best_minima) else "r" for v in local_solutions[var][scene_ind]]
            plt.scatter([var] * num_local_solves, [min(v[3].real) for v  in local_solutions[var][scene_ind]], color = colors)
    plt.yscale("symlog")
    plt.ylabel("Log of minimum eigenvalue from local solver")
    plt.xlabel("Pixel space gaussian measurement variance")
    plt.savefig(os.path.join(exp_dir, f"eig_plot.png"))
    plt.show()
    plt.close("all")

    heights = [num_certified_per_var[k]/num_instances for k in var_list]
    plt.bar([str(v) for v in var_list], heights)
    plt.ylabel("Percentage of 'globally optimal' solutions that were certified")
    plt.xlabel("Pixel space gaussian measurement variance")
    plt.savefig(os.path.join(exp_dir, f"certified_plot.png"))
    plt.show()
    print(var_list)
    print(heights)

if __name__ == "__main__":
    main()