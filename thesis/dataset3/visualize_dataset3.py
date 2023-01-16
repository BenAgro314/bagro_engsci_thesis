#%% imports
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pylgmath.so3.operations import hat

import sys
sys.path.append("..")

import plotting
from sim import Camera, World, render_camera_points

#%% load dataset

dataset3 = scipy.io.loadmat('dataset3.mat')

#%% visualize ground truth

# F_i: inertial frame
# F_v: vehicle frame
# F_c: camera frame

theta_vk_i = dataset3["theta_vk_i"] # a 3xK matrix
r_i_vk_i = dataset3["r_i_vk_i"] # a 3xK matrix where the kth column is the groundtruth position of the camera at timestep k

y_k_j = dataset3["y_k_j"] # 4 x K x 20 array of observations. All components of y_k_j(:, k, j) will be -1 if the observation is invalid
y_var = dataset3["y_var"] # 4 x 1 matrix of computed variances based on ground truth stereo measurements
rho_i_pj_i = dataset3["rho_i_pj_i"] # a 3x20 matrix where the jth column is the poisition of feature j

# camera to vehicle
C_c_v = dataset3["C_c_v"] # 3 x 3 matrix giving rotation from vehicle frame to camera frame
rho_v_c_v = dataset3["rho_v_c_v"]

# intrinsics
fu = dataset3["fu"]
fv = dataset3["fv"]
cu = dataset3["cu"]
cv = dataset3["cv"]
b = dataset3["b"]

# %% Motion model


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for landmark_pt in rho_i_pj_i.T:
    ax.scatter3D(landmark_pt[0], landmark_pt[1], landmark_pt[2], color = 'r')


list_T_vk_i = np.stack([np.eye(4)] * theta_vk_i.shape[1], axis = 0)
list_T_ck_i = np.stack([np.eye(4)] * theta_vk_i.shape[1], axis = 0)
list_T_i_ck = np.stack([np.eye(4)] * theta_vk_i.shape[1], axis = 0)

T_c_v = np.eye(4)
T_c_v[:3, :3] = C_c_v
T_c_v[:3, -1:] = -C_c_v @ rho_v_c_v

for k, psi in enumerate(theta_vk_i.T):
    psi = psi.reshape(3, 1)
    psi_mag = np.linalg.norm(psi)
    C_vk_i = np.cos(psi_mag) * np.eye(3) + ( 1 - np.cos(psi_mag) ) * (psi / psi_mag) @ (psi / psi_mag).T - np.sin(psi_mag) * hat(psi / psi_mag)
    T_vk_i = np.eye(4)
    T_vk_i[:3, :3] = C_vk_i
    T_vk_i[:3, -1] = - C_vk_i @ r_i_vk_i[:, k]
    T_ck_i = T_c_v @ T_vk_i

    list_T_vk_i[k] = T_vk_i
    list_T_ck_i[k] = T_ck_i
    list_T_i_ck[k] = np.linalg.inv(T_ck_i)


# for k, T_i_ck in enumerate(list_T_i_ck):
#     if k % 100 != 0:
#         continue
#     plotting.add_coordinate_frame(T_i_ck, ax, "$\mathcal{F}" + f"_{k}$", size = 0.5)

ax.plot3D(list_T_i_ck[:, 0, -1], list_T_i_ck[:, 1, -1], list_T_i_ck[:, 2, -1])
plt.show()

# %%
ind = 800

#Camera(fu, fv, cu, cv, b, np.diag(y_var), None) # is this noise correct?
no_noise_camera = Camera(fu, fv, cu, cv, b, 0 * np.eye(4), fov_depth_range=(0,0), fov_phi_range=(0,0)) # is this noise correct?
# 3 x 20
p_w = rho_i_pj_i.T[:, :, None]
p_w = np.concatenate((p_w, np.ones_like(p_w[:, 0:1, :])), axis = 1)
world = World(no_noise_camera, None, p_w.shape[0], list_T_i_ck[ind], p_w)
fig, ax, colors = world.render()

y = no_noise_camera.take_picture(list_T_i_ck[ind], p_w)
camfig, (l_ax, r_ax) = render_camera_points(y, colors)

# 20 x 4 x 1, all -1 if invalid
ys_with_invalid = dataset3["y_k_j"][:, ind, :].T[:, :, None]
mask = ~((ys_with_invalid.squeeze(-1) == -1).all(axis = 1))
y = ys_with_invalid[mask]
colors = colors[mask]

camfig, (l_ax, r_ax) = render_camera_points(y, colors)

#%%

import pickle as pkl
# inspect experiment results 

f = open("/Users/benagro/bagro_engsci_thesis/outputs/2023-01-02-19:02:59/metrics.pkl", "rb")
metrics = pkl.load(f)
f.close()

#%%

vars = list(set(m["noise_var"] for m in metrics))
vars.sort()
scene_inds = [m["scene_ind"] for m in metrics]
min_costs = {}
local_counts = {v: 0 for v in vars}
iter_sdp_counts = {v: 0 for v in vars}
totals = {v: 0 for v in vars}
for var in vars:
    min_costs[var] = {}
    for scene_ind in scene_inds:
        min_costs[var][scene_ind] = min([m["local_solution"].cost for m in metrics if (m["noise_var"] == var and m["scene_ind"] == scene_ind)])

#%%
import numpy as np
from iterative_sdp import iterative_sdp_solution

non_optimal = []

for m in metrics:
    var = m["noise_var"]
    iterative_sdp_cost = m["iterative_sdp_solution"].cost
    scene_ind = m["scene_ind"]
    if not np.isclose(iterative_sdp_cost, min_costs[var][scene_ind]):
        non_optimal.append(m)

#%%

isclose_num = 0

for i in range(len(non_optimal)):
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    m = non_optimal[i]
    #plotting.add_coordinate_frame(m["problem"].T_wc, ax, "$\mathfrak{F}_c$")
    ##plotting.add_coordinate_frame(np.linalg.inv(m["iterative_sdp_solution"].T_cw), ax, "$\mathfrak{F}_{soln}$")
    #y = m["problem"].y
    #p_w = m["problem"].p_w
    #num_landmarks = len(y)
    #colors = np.random.rand(num_landmarks, 3)
    #colors = np.concatenate((colors, np.ones_like(colors[:, 0:1])), axis = 1)
    #for i, p in enumerate(p_w):
    #    ax.scatter3D(p[0], p[1], p[2], color = colors[i])

    #world_limits = ax.get_w_lims()
    #ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
    soln = iterative_sdp_solution(m["problem"], m["problem"].T_init, max_iters = 10, return_X = False, log = False, min_update_norm = 0.02, max_num_tries = 10)
    isclose_num += int(np.isclose(soln.cost, min_costs[m["noise_var"]][m["scene_ind"]]))

#%%
print(isclose_num)

#%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
m = non_optimal[15]
plotting.add_coordinate_frame(m["problem"].T_wc, ax, "$\mathfrak{F}_c$")
#plotting.add_coordinate_frame(np.linalg.inv(m["iterative_sdp_solution"].T_cw), ax, "$\mathfrak{F}_{soln}$")
y = m["problem"].y
p_w = m["problem"].p_w
num_landmarks = len(y)
colors = np.random.rand(num_landmarks, 3)
colors = np.concatenate((colors, np.ones_like(colors[:, 0:1])), axis = 1)
for i, p in enumerate(p_w):
    ax.scatter3D(p[0], p[1], p[2], color = colors[i])

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
soln = iterative_sdp_solution(m["problem"], m["problem"].T_init, max_iters = 10, return_X = False, log = False, min_update_norm = 0.02, max_num_tries = 10)
#isclose_num += int(np.isclose(soln.cost, min_costs[m["noise_var"]][m["scene_ind"]]))