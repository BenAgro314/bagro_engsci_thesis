#%% imports

import pickle
import numpy as np
from thesis.simulation.sim import render_camera_points
import thesis.visualization.plotting as plotting
import os


#%% 

dir_path = "/Users/benagro/bagro_engsci_thesis/outputs/2022-11-27-08:33:57"

f = open(os.path.join(dir_path, "metrics.pkl"), "rb")
metrics = pickle.load(f)

unsolved = [m for m in metrics if not m["iterative_sdp_solution"].solved]
solved = [m for m in metrics if m["iterative_sdp_solution"].solved]

#%%

#for i in range(len(unsolved)):
for i in range(len(unsolved)):
    problem = unsolved[i]["problem"]
    world = unsolved[i]["world"]

    world.T_wc = problem.T_wc
    world.p_w = problem.p_w

    fig, ax, colors = world.render()
    plotting.add_coordinate_frame(np.linalg.inv(problem.T_init), ax, "$\mathfrak{F}_{init}$")
    fig.savefig(os.path.join(dir_path, f"unsolved/unsolved_world_{i}.png"))

    # Generative camera model 
    y = world.cam.take_picture(np.linalg.inv(problem.T_init), world.p_w)
    camfig, (l_ax, r_ax) = render_camera_points(y, colors)
    camfig.savefig(os.path.join(dir_path, f"unsolved/unsolved_cam_{i}.png"))

for i in range(len(solved)):
    problem = solved[i]["problem"]
    world = solved[i]["world"]

    world.T_wc = problem.T_wc
    world.p_w = problem.p_w

    fig, ax, colors = world.render()
    plotting.add_coordinate_frame(np.linalg.inv(problem.T_init), ax, "$\mathfrak{F}_{init}$")
    fig.savefig(os.path.join(dir_path, f"solved/solved_world_{i}.png"))

    # Generative camera model 
    y = world.cam.take_picture(np.linalg.inv(problem.T_init), world.p_w)
    camfig, (l_ax, r_ax) = render_camera_points(y, colors)
    camfig.savefig(os.path.join(dir_path, f"solved/solved_cam_{i}.png"))   
