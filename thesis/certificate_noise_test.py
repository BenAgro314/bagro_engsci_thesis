import os 
from typing import Dict, Tuple, List

from thesis.experiments import run_experiment
import numpy as np
from thesis.relaxations.certificate import run_certificate
import thesis.visualization.plotting as plotting
import thesis.simulation.sim as sim
from thesis.solvers.local_solver import stereo_localization_gauss_newton

def metrics_fcn(problem):
    datum = {}
    solution = stereo_localization_gauss_newton(problem, log = False, max_iters = 100)
    datum["local_solution"] = solution
    if solution.T_cw is not None:
        certificate = run_certificate(problem, solution)
        datum["certificate"] = certificate
    return datum

def main():

    var_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]#, 3, 5, 7, 9, 10]
    num_problem_instances = 5
    num_landmarks = 20
    num_local_solve_tries = 100

    cam = sim.Camera(
        f_u = 484.5,
        f_v = 484.5,
        c_u = 322,
        c_v = 247,
        b = 0.24,
        R = 0 * np.eye(4), # co-variance matrix for image-space noise
        fov_phi_range = (-np.pi / 12, np.pi / 12),
        fov_depth_range = (0.2, 3),
    )

    p_wc_extent = np.array([[3], [3], [0]])

    r0 = np.zeros((3, 1))
    gamma_r = 0 #1e-1

    metrics = []

    metrics, exp_dir = run_experiment(metrics_fcn, var_list, num_problem_instances, num_landmarks, num_local_solve_tries, cam, p_wc_extent, r0 = r0, gamma_r = gamma_r)
    plotting.plot_minimum_eigenvalues(metrics, os.path.join(exp_dir, "min_eigs_plt.png"))


if __name__ == "__main__":
    main()