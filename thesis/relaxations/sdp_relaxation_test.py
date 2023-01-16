#%%

# Imports
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from thesis.simulation.sim import render_camera_points, World, Camera
from thesis.solvers.local_solver import projection_error, StereoLocalizationProblem
from thesis.relaxations.sdp_relaxation import (
    build_general_SDP_problem,
    build_cost_matrix,
    build_rotation_constraint_matrices,
    build_measurement_constraint_matrices,
    build_parallel_constraint_matrices,
    extract_solution_from_X,
)

#%% make problem

cam = Camera(
    f_u = 160, # focal length in horizontal pixels
    f_v = 160, # focal length in vertical pixels
    c_u = 320, # pinhole projection in horizontal pixels
    c_v = 240, # pinhole projection in vertical pixels
    b = 0.25, # baseline (meters)
    R = 0 * np.eye(4), # co-variance matrix for image-space noise
    fov_phi_range = (-np.pi / 12, np.pi / 12),
    fov_depth_range = (0.2, 3),
)

world = World(
    cam = cam,
    p_wc_extent = np.array([[3], [3], [0]]),
    num_landmarks = 5,
)

world.clear_sim_instance()
world.make_random_sim_instance()
fig, ax, colors = world.render()

# Generative camera model 
y = cam.take_picture(world.T_wc, world.p_w)
camfig, (l_ax, r_ax) = render_camera_points(y, colors)

W =  np.eye(4)
r0 = np.zeros((3, 1))
gamma_r = 0
T_op = np.eye(4)

Ws = np.zeros((world.num_landmarks, 4, 4))
for i in range(world.num_landmarks):
    Ws[i] = W


# local solver

problem = StereoLocalizationProblem(world.T_wc, world.p_w, cam.M(), W, y, r_0 = r0, gamma_r = gamma_r)
problem.T_init = T_op
solution = local_solver.stereo_localization_gauss_newton(problem, log = True)
T_op = solution.T_cw
local_minima = solution.cost
print("Estimate:\n", T_op)
print("Ground Truth:\n", np.linalg.inv(world.T_wc))

#%% Iterative SDP

n = 13 + 3 * world.num_landmarks
Q = build_cost_matrix(world.num_landmarks, y, Ws, cam.M(), r0, gamma_r)

As = []
bs = []

# rotation matrix
rot_matrix_As, bs = build_rotation_constraint_matrices()
for rot_matrix_A in rot_matrix_As:
    A = np.zeros((n, n))
    A[:9, :9] = rot_matrix_A
    As.append(A)

# homogenization variable
A = np.zeros((n, n))
A[-1, -1] = 1
As.append(A)
bs.append(1)

# measurments
A_measure, b_measure = build_measurement_constraint_matrices(world.p_w)
As += A_measure
bs += b_measure

# redundant constraints
if True:
    A_par, b_par = build_parallel_constraint_matrices(world.p_w)
    As += A_par
    bs += b_par

eps = 1e-10 # try 1e-10
mosek_params = {
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps,
}

Q_new = Q / np.mean(np.abs(Q)) # improve numerics
prob, X = build_general_SDP_problem(Q_new, As, bs)
prob.solve(solver=cp.MOSEK, verbose = False)# , mosek_params = mosek_params)

print("The optimal value from the SDP is", prob.value)
print("The optimal value from the local solver is", local_minima)
X = X.value

print("Ground Truth:\n", np.linalg.inv(world.T_wc))
T_sdp = extract_solution_from_X(X)
print("SDP Solution:\n", T_sdp)
cost = projection_error(y, T_sdp, cam.M(), world.p_w, W)
print("SDP Solution Cost:", cost)
print("Local Solution Cost:", local_minima[0][0])

eig_values, eig_vectors = np.linalg.eig(X)
plt.close("all")
eig_values
plt.scatter(range(len(eig_values)), eig_values)
plt.ylabel("$\lambda$")
plt.savefig("eigs.png")