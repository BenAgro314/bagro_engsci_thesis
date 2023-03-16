import os 
import sys
import numpy as np

from thesis.datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample, StereoLocalizationDatasetConfig
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.relaxations.certificate import run_certificate
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import plot_minimum_eigenvalues

def metrics_fcn(example: StereoLocalizationExample):
    problem = example.problem
    datum = {}
    solution = stereo_localization_gauss_newton(problem, log = False, max_iters = 100)
    datum["local_solution"] = solution
    datum["noise_var"] = np.mean(np.diag(example.camera.R))
    datum["scene_ind"] = example.example_id
    #assert np.all(example.camera.R == datum["noise_var"] * np.eye(4)), f"{example.camera.R}"
    if solution.solved:
        certificate = run_certificate(problem, solution)
        datum["certificate"] = certificate
    return datum

def main(dataset_name: str):
    dataset = StereoLocalizationDataset.from_pickle(os.path.join(get_data_dir_path(), dataset_name + ".pkl"))
    metrics, exp_dir = run_experiment(dataset=dataset, metrics_fcn=metrics_fcn)
    plot_minimum_eigenvalues(metrics, os.path.join(exp_dir, "min_eigs"))

if __name__ == "__main__":
    assert len(sys.argv) == 2, "python certificate_noise_test.py <dataset_name>"
    dataset_name = sys.argv[1]
    main(dataset_name)