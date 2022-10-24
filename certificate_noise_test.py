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
from certificate import run_certificate
import plotting
import sim
import local_solver

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
        instances.append(local_solver.StereoLocalizationProblem(world.T_wc, world.p_w, cam.M()))
    return instances

def main():
    """Experiment Outline:
        1. Iterate over noise levels and number of points 
    """

    exp_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, f"outputs/{exp_time}") 
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    var_list = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
    num_problem_instances = 1#5
    num_landmarks = 10
    num_local_solve_tries = 20 #100

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

    r0 = np.zeros((3, 1))
    gamma_r = 0
    W = np.eye(4)

    world = sim.World(
        cam = cam,
        p_wc_extent = p_wc_extent,
        num_landmarks = num_landmarks,
    )

    metrics = {}

    for var in var_list:
        print(f"Noise Variance: {var}")
        metrics[var] = {}

        world.cam.R = var * np.eye(4)
        for scene_ind in range(num_problem_instances):
            print(f"Scene ind: {scene_ind}")
            metrics[var][scene_ind] = []


            problem = instances[scene_ind]
            problem.y = world.cam.take_picture(problem.T_wc, problem.p_w)
            problem.W = W
            problem.r_0 = r0
            problem.gamma_r = gamma_r

            for _ in tqdm.tqdm(range(num_local_solve_tries)):
                # local solution
                datum = {}
                T_op = sim.generate_random_T(p_wc_extent)
                solution = local_solver.stereo_localization_gauss_newton(problem, T_op, log = False, max_iters = 100)
                datum["problem"] = problem
                datum["solution"] = solution

                if solution.T_cw is not None:
                    certificate = run_certificate(problem, solution)
                    datum["certificate"] = certificate
                metrics[var][scene_ind].append(datum)

    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    plotting.plot_minimum_eigenvalues(metrics, os.path.join(exp_dir, "min_eigs_plt.png"))


if __name__ == "__main__":
    main()