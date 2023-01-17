import os 
import sys
import numpy as np
from solvers.global_sdp import global_sdp_solution
from solvers.iterative_sdp import iterative_sdp_solution

from thesis.datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import plot_percent_succ_vs_noise

RECORD_HISTORY=False

def metrics_fcn(example: StereoLocalizationExample, num_tries = 100):
    problem = example.problem
    datum = {}
    datum["noise_var"] = example.camera.R[0, 0]
    datum["scene_ind"] = example.example_id
    assert np.all(example.camera.R == datum["noise_var"] * np.eye(4)), f"{example.camera.R}"
    mosek_params = {}
    local_solution = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, num_tries = num_tries, record_history=RECORD_HISTORY)
    iter_sdp_soln = iterative_sdp_solution(problem, problem.T_init, max_iters = 5, min_update_norm = 0.02, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY)
    global_sdp_soln = global_sdp_solution(problem, problem.T_init, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY)
    datum["local_solution"] = local_solution
    datum["iterative_sdp_solution"] = iter_sdp_soln
    datum["global_sdp_solution"] = global_sdp_soln

    return datum

def main(dataset_name: str):
    dataset = StereoLocalizationDataset.from_pickle(os.path.join(get_data_dir_path(), dataset_name))
    metrics, exp_dir = run_experiment(dataset=dataset, metrics_fcn=metrics_fcn)
    plot_percent_succ_vs_noise(metrics, os.path.join(exp_dir, "local_vs_iterative_bar.png"))

if __name__ == "__main__":
    assert len(sys.argv) == 2, "python certificate_noise_test.py <dataset_name>"
    dataset_name = sys.argv[1]
    main(dataset_name)