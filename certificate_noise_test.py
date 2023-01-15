import os 
import pickle
from datetime import datetime
from typing import Dict, Tuple, List

from experiments import run_experiment
import numpy as np
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

def metrics_fcn(problem):
    datum = {}
    solution = local_solver.stereo_localization_gauss_newton(problem, log = False, max_iters = 100)
    datum["local_solution"] = solution
    if solution.T_cw is not None:
        certificate = run_certificate(problem, solution)
        datum["certificate"] = certificate
    return datum

def main():

    var_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9, 10]
    num_problem_instances = 5
    num_landmarks = 4
    num_local_solve_tries = 100

    cam = sim.Camera(
        f_u = 160, # focal length in horizontal pixels
        f_v = 160, # focal length in vertical pixels
        c_u = 320, # pinhole projection in horizontal pixels
        c_v = 240, # pinhole projection in vertical pixels
        b = 0.25, # baseline (meters)
        R = 0 * np.eye(4), # co-variance matrix for image-space noise
        fov = np.array([[-1,1], [-1, 1], [2, 5]])
    )

    p_wc_extent = np.array([[3], [3], [0]])

    r0 = np.zeros((3, 1))
    gamma_r = 0 #1e-1

    metrics = []

    metrics, exp_dir = run_experiment(metrics_fcn, var_list, num_problem_instances, num_landmarks, num_local_solve_tries, cam, p_wc_extent, r0 = r0, gamma_r = gamma_r)
    plotting.plot_minimum_eigenvalues(metrics, os.path.join(exp_dir, "min_eigs_plt.png"))


if __name__ == "__main__":
    main()