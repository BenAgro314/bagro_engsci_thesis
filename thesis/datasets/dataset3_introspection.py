#%%
import os 
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/benagro/bagro_engsci_thesis/")
sys.path.append("/Users/benagro/bagro_engsci_thesis/thesis/")
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.iterative_sdp import iterative_sdp_solution

from datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample, StereoLocalizationDatasetConfig
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import plot_percent_succ_vs_noise, add_coordinate_frame
import numpy as np
import time
#%%

dataset_name = "starry_nights"
dataset = StereoLocalizationDataset.from_pickle(os.path.join(get_data_dir_path(), dataset_name + ".pkl"))
#%%
example = dataset[5660]
world = example.world

eps = 1e-5 # try 1e-10
mosek_params = {
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps,
}
iter_soln = iterative_sdp_solution(example.problem, example.problem.T_init, max_iters = 5, return_X = False, log = True, min_update_norm = 0.2, max_num_tries = 1, refine = False, mosek_params = mosek_params)

global_soln = global_sdp_solution(example.problem, return_X = False)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
add_coordinate_frame(example.problem.T_wc, ax, "$\mathfrak{F}_s$")
add_coordinate_frame(example.problem.T_init, ax, "$\mathfrak{F}_{init}$")
add_coordinate_frame(np.linalg.inv(iter_soln.T_cw), ax, "$\mathfrak{F}_{iter}$")
add_coordinate_frame(np.linalg.inv(global_soln.T_cw), ax, "$\mathfrak{F}_{global}$")
#plotting.add_coordinate_frame(np.linalg.inv(m["iterative_sdp_solution"].T_cw), ax, "$\mathfrak{F}_{soln}$")
y = example.problem.y
p_w = example.problem.p_w
num_landmarks = len(y)
colors = np.random.rand(num_landmarks, 3)
colors = np.concatenate((colors, np.ones_like(colors[:, 0:1])), axis = 1)
for i, p in enumerate(p_w):
    ax.scatter3D(p[0], p[1], p[2], color = colors[i])

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
# %%
