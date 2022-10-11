from typing import Dict, Optional

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

def stereo_sim_solve_certify(world: sim.World, r0: np.array, gamma_r: float, filepath: Optional[str] = None):
    world.clear_sim_instance()
    world.make_random_sim_instance()

    # Generative camera model 
    y = world.cam.take_picture(world.T_wc, world.p_w)

    # local solution
    T_op = np.eye(4)
    W = np.eye(4)
    T_op, local_minima = local_solver.stereo_localization_gauss_newton(
        T_op, y, world.p_w, W, world.cam.M(), r_0 = r0, gamma_r = gamma_r, log = False
    )

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


def main():
    """Experiment Outline:
        1. Iterate over noise levels and number of points 
    """

    iters_per_noise = 100

    for var in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        print(f"Noise Variance: {var}")
        for i in range(iters_per_noise):
            if i % 1 == 0:
                print(f"Iteration [{i}/{iters_per_noise}]")
            cam = sim.Camera(
                f_u = 100, # focal length in horizonal pixels
                f_v = 100, # focal length in vertical pixels
                c_u = 50, # pinhole projection in horizonal pixels
                c_v = 50, # pinhold projection in vertical pixels
                b = 0.2, # baseline (meters)
                R = 1 * np.eye(4), # covarience matrix for image-space noise
                fov = np.array([[-1,1], [-1, 1], [2, 5]])
            )
            world = sim.World(
                cam = cam,
                p_wc_extent = np.array([[3], [3], [0]]),
                num_landmarks = 5,
            )
            stereo_sim_solve_certify(world, r0 = np.zeros((3, 1)), gamma_r = 0)

if __name__ == "__main__":
    main()