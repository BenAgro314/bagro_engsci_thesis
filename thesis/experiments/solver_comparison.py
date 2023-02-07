import os 
import sys
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.iterative_sdp import iterative_sdp_solution

from thesis.datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample, StereoLocalizationDatasetConfig
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import plot_percent_succ_vs_noise, plot_select_solutions_history, plot_solution_time_vs_num_landmarks
import numpy as np
import time

RECORD_HISTORY=True

global_sdp_solved_example_ids = {}

def metrics_fcn(example: StereoLocalizationExample, num_tries = 100):
    problem = example.problem
    example_id = example.example_id
    datum = {}
    datum["noise_var"] = example.camera.R[0, 0]
    datum["scene_ind"] = example_id
    datum["example"] = example

    start = time.process_time()
    local_solution = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, num_tries = num_tries, record_history=RECORD_HISTORY)
    datum["local_solution_time"] = time.process_time() - start

    start = time.process_time()
    eps = 1e-5
    mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps,
    }
    iter_sdp_soln = iterative_sdp_solution(problem, problem.T_init, max_iters = 5, min_update_norm = 0.2, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY)
    datum["iterative_sdp_solution_time"] = time.process_time() - start

    if example_id not in global_sdp_solved_example_ids:
        mosek_params = {}
        start = time.process_time()
        global_sdp_soln = global_sdp_solution(problem, return_X = False, mosek_params=mosek_params, record_history=RECORD_HISTORY)
        datum["global_sdp_solution_time"] = time.process_time() - start
        global_sdp_solved_example_ids[example_id] = global_sdp_soln, datum["global_sdp_solution_time"]
        datum["global_sdp_solution"] = global_sdp_soln
    else:
        datum["global_sdp_solution"], datum["global_sdp_solution_time"] = global_sdp_solved_example_ids[example_id]

    datum["local_solution"] = local_solution
    datum["iterative_sdp_solution"] = iter_sdp_soln

    return datum

def main(dataset_name: str):
    dataset = StereoLocalizationDataset.from_pickle(os.path.join(get_data_dir_path(), dataset_name + ".pkl"))
    metrics, exp_dir = run_experiment(dataset=dataset, metrics_fcn=metrics_fcn)
    
    #import pickle as pkl
    #f = open("/Users/benagro/bagro_engsci_thesis/thesis/experiments/outputs/2023-01-20-08:09:40/metrics.pkl", "rb")
    #metrics = pkl.load(f)
    #exp_dir = "/Users/benagro/bagro_engsci_thesis/thesis/experiments/outputs/2023-01-20-08:09:40" 

    plot_percent_succ_vs_noise(metrics, os.path.join(exp_dir, "solver_comparison_success"))
    plot_solution_time_vs_num_landmarks(metrics, os.path.join(exp_dir, "solver_comparison_time"))
    if RECORD_HISTORY:
        plot_select_solutions_history(metrics, exp_dir, num_per_noise = 3)

    local_solution_times = []
    iter_sdp_solution_times = []
    global_sdp_solution_times = []

    for m in metrics:
        local_solution_times.append(m["local_solution_time"])
        iter_sdp_solution_times.append(m["iterative_sdp_solution_time"])
        global_sdp_solution_times.append(m["global_sdp_solution_time"])

    local_solution_times = np.array(local_solution_times)
    iter_sdp_solution_times = np.array(iter_sdp_solution_times)
    global_sdp_solution_times = np.array(global_sdp_solution_times)

    print(f"Local solver average time: {np.mean(local_solution_times)} +/- {np.std(local_solution_times)}")
    print(f"Iterative SDP average time: {np.mean(iter_sdp_solution_times)} +/- {np.std(iter_sdp_solution_times)}")
    print(f"Global SDP average time: {np.mean(global_sdp_solution_times)} +/- {np.std(global_sdp_solution_times)}")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "python certificate_noise_test.py <dataset_name>"
    dataset_name = sys.argv[1]
    main(dataset_name)