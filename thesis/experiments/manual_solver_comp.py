#%%
from copy import deepcopy
import os 
import sys

from matplotlib import pyplot as plt

sys.path.append("/Users/benagro/bagro_engsci_thesis")
from thesis.experiments.utils import StereoLocalizationProblem
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.iterative_sdp import iterative_sdp_solution
from thesis.simulation.sim import Camera, World, generate_random_T, generate_stereo_camera_noise, render_camera_points

from thesis.datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample, StereoLocalizationDatasetConfig
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import add_coordinate_frame, plot_percent_succ_vs_noise, plot_select_solutions_history, plot_solution_time_vs_num_landmarks
import numpy as np
import time

#%% Construct problem

num_landmarks = 3

# make camera
cam = Camera(
    f_u = 484.5,
    f_v = 484.5, 
    c_u = 322,
    c_v = 247,
    b = 0.24,
    R = 0 * np.eye(4),
    fov_phi_range = (-np.pi/12, np.pi/12),
    fov_depth_range = (0.5, 5),
)
world = World(
    cam = cam,
    p_wc_extent = np.array([[3], [3], [0]]),
    num_landmarks = num_landmarks,
)
world.clear_sim_instance()
world.T_wc = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
min_phi, max_phi = world.cam.fov_phi_range
assert min_phi < max_phi
min_rho, max_rho = world.cam.fov_depth_range
assert min_rho < max_rho
phi = np.random.rand(world.num_landmarks, 1) * (max_phi - min_phi) + min_phi
theta = np.random.rand(world.num_landmarks, 1) * 2*np.pi
rho = np.random.rand(world.num_landmarks, 1) * (max_rho - min_rho) + min_rho
# rho[-1] = 100
#rho = np.array([
#    [20],
#    [10],
#    [5],
#    [1],
#])
z = np.cos(phi) * rho
x = np.sin(phi) * np.cos(theta) * rho
y = np.sin(phi) * np.sin(theta) * rho
homo_p_c = np.concatenate((x, y, z, np.ones_like(x)), axis = 1) # (N, 4, 1)
world.p_w = world.T_wc @ homo_p_c[:, :, None] # (N, 4, 1), world frame points


#world.make_random_sim_instance()
fig, ax, colors = world.render()

# Generative camera model 
y = cam.take_picture(world.T_wc, world.p_w)
R = np.eye(4) * 0.5
dy = generate_stereo_camera_noise(R, size = y.shape[0])[:, :, None]
#dy[-1] = 100 * (2 * np.random.rand(4, 1) - 1)
#print(dy)
y += dy
camfig, (l_ax, r_ax) = render_camera_points(y, colors)


#%% global minima
W =  np.eye(4)
r0 = np.zeros((3, 1))
gamma_r = 0

global_min_cost = np.inf
global_min_T = None

problem = StereoLocalizationProblem(world.T_wc, world.p_w, cam.M(), W, y, r_0 = r0, gamma_r = gamma_r, T_init = generate_random_T(world.p_wc_extent))

for i in range(20):
    T_init = generate_random_T(world.p_wc_extent)
    p_tmp = deepcopy(problem)
    p_tmp.T_init = T_init
    solution = stereo_localization_gauss_newton(p_tmp, log = False)
    T_op = solution.T_cw
    local_minima = solution.cost[0][0]
    if solution.solved and local_minima < global_min_cost:
        global_min_cost = local_minima
        global_min_T = T_op


#%%
num_tries = 1
RECORD_HISTORY = False
mosek_params = {}

local_solution = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, num_tries = num_tries, record_history=RECORD_HISTORY)
iter_sdp_soln = iterative_sdp_solution(problem, problem.T_init, max_iters = 10, min_update_norm = 1e-10, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY, refine=False, log = False)
iter_sdp_soln_refine = iterative_sdp_solution(problem, problem.T_init, max_iters = 20, min_update_norm = 1e-5, return_X = False, mosek_params=mosek_params, max_num_tries = num_tries, record_history=RECORD_HISTORY, refine=True, log=False)
global_sdp_soln = global_sdp_solution(problem, return_X = False, mosek_params=mosek_params, record_history=RECORD_HISTORY, refine=False)
global_sdp_soln_refine = global_sdp_solution(problem, return_X = False, mosek_params=mosek_params, record_history=RECORD_HISTORY, refine=True)

print(f"Global minima: {global_min_cost}")
#add_coordinate_frame(np.linalg.inv(global_min_T), ax, "$\mathfrak{F}_{global_min}$")
print(f"Local solution cost: {local_solution.cost[0][0]}")
#add_coordinate_frame(np.linalg.inv(local_solution.T_cw), ax, "$\mathfrak{F}_{local}$")
print(f"Iter SDP cost: {iter_sdp_soln.cost}")
print(f"Refined Iter SDP refine: {iter_sdp_soln_refine.cost[0][0]}")
print(f"Global SDP cost: {global_sdp_soln.cost}")
print(f"Refined Global SDP cost: {global_sdp_soln_refine.cost[0][0]}")
#add_coordinate_frame(np.linalg.inv(global_sdp_soln.T_cw), ax, "$\mathfrak{F}_{global_SDP}$")
#%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
add_coordinate_frame(np.linalg.inv(global_min_T), ax, "$\mathfrak{F}_{global_min}$")
world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
add_coordinate_frame(np.linalg.inv(T_init), ax, "$\mathfrak{F}_{init}$")
add_coordinate_frame(np.linalg.inv(iter_sdp_soln.T_cw), ax, "$\mathfrak{F}_{iter}$")
add_coordinate_frame(np.linalg.inv(global_sdp_soln.T_cw), ax, "$\mathfrak{F}_{global_SDP}$")