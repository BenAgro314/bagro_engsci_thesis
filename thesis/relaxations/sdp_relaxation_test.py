#%%

# Imports
import sys
from ncpol2sdpa import *

sys.path.append("/home/agrobenj/bagro_engsci_thesis")
sys.path.append("/home/agrobenj/bagro_engsci_thesis/thesis/")
import cvxpy as cp
from thesis.relaxations.sdp_relaxation import build_cost_matrix_v2, build_general_SDP_problem, build_homo_constraint, build_measurement_constraint_matrices_v2, build_parallel_constraint_matrices, build_rotation_constraint_matrices, extract_solution_from_X, build_redundant_rotation_constraint_matrices
import numpy as np
import matplotlib.pyplot as plt
from thesis.solvers.unknown_scale_sdp import unknown_scale_sdp
from thesis.simulation.sim import render_camera_points, World, Camera
import thesis.solvers.local_solver as local_solver
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.local_solver import projection_error, StereoLocalizationProblem
from thesis.relaxations.sdp_relaxation import (
    build_general_SDP_problem,
    build_cost_matrix,
    build_rotation_constraint_matrices,
    build_measurement_constraint_matrices,
    build_parallel_constraint_matrices,
    extract_solution_from_X,
)
import tikzplotlib
import scipy
import scipy.io

#%% make problem

cam = Camera(
    f_u = 484.5,
    f_v = 484.5,
    c_u = 322,
    c_v = 247,
    b = 0.24,
    R = 0.1 * np.eye(4),
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

#%%

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


#%% global SDP
eps = 1e-10
mosek_params = {
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps,
}

X_sdp = global_sdp_solution(problem, return_X = True, mosek_params=mosek_params, record_history=False, redundant_constraints = False, log = True)

print("Ground Truth:\n", np.linalg.inv(world.T_wc))
T_sdp = extract_solution_from_X(X_sdp)
print("SDP Solution:\n", T_sdp)
cost = projection_error(y, T_sdp, cam.M(), world.p_w, W)
print("SDP Solution Cost:", cost)
print("Local Solution Cost:", local_minima[0][0])

eig_values, eig_vectors = np.linalg.eig(X_sdp)
plt.close("all")
eig_values
plt.scatter(range(len(eig_values)), eig_values)
plt.ylabel("Eigenvalues of $\mathbf{X}$ ($\lambda$)")
#plt.savefig("eigs.png")
tikzplotlib.save("global_sdp_eigs.tex")

#%%
X = global_sdp_solution(problem, return_X = True, mosek_params=mosek_params, record_history=False, redundant_constraints = True, log = True)

#print("Ground Truth:\n", np.linalg.inv(world.T_wc))
#T_sdp = extract_solution_from_X(X)
#print("SDP Solution:\n", T_sdp)
#cost = projection_error(y, T_sdp, cam.M(), world.p_w, W)
T_sdp = extract_solution_from_X(X)
cost = projection_error(y, T_sdp, cam.M(), world.p_w, W)
print("SDP With Redundant Constraints Solution Cost:", cost)
print("Local Solution Cost:", local_minima[0][0])


eig_values, eig_vectors = np.linalg.eig(X)
plt.close("all")
eig_values
plt.scatter(range(len(eig_values)), eig_values)
plt.ylabel("Eigenvalues of $\mathbf{X}$ ($\lambda$)")
#plt.savefig("eigs.png")
tikzplotlib.save("global_sdp_eigs_redundant.tex")

#%% Laserrres

num_landmarks = problem.y.shape[0]
Ws = np.zeros((num_landmarks, 4, 4))
for i in range(num_landmarks):
    Ws[i] = problem.W

Q = build_cost_matrix_v2(num_landmarks, problem.y, Ws, problem.M, problem.r_0, problem.gamma_r)
Q = Q / np.mean(np.abs(Q)) # improve numerics 
As = []
bs = []

d = 13 + 3 * num_landmarks
assert Q.shape == (d, d)

# rotation matrix
As_rot, bs = build_rotation_constraint_matrices()
for A_rot in As_rot:
    A = np.zeros((d, d))
    A[:9, :9] = A_rot
    As.append(A)


# homogenization variable
A, b = build_homo_constraint(num_landmarks)
As.append(A)
bs.append(b)

# measurements
A_measure, b_measure = build_measurement_constraint_matrices_v2(problem.p_w)
As += A_measure
bs += b_measure

mdict = {
    "Q": Q,
    "As": As,
    "bs": bs
}

scipy.io.savemat("/home/agrobenj/bagro_engsci_thesis/thesis/matlab/sdp_test.mat", mdict)

#%%

n_vars = Q.shape[0]
level = 1
x = generate_variables('x', n_vars)
obj = np.dot(x, np.dot(Q, np.transpose(x)))
#inequalities = [-x[1]**2 + x[1] + 0.5>=0]
equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b for A, b in zip(As, bs)]
sdp = SdpRelaxation(x)
print(f"Build optimization problem!")
sdp.get_relaxation(level, objective=obj, equalities=equalities)
print(f"Relaxed problem!")
print(f"Solving problem")
sdp.solve(solver = 'mosek')
print(f"Done solving problem")
X = sdp.x_mat[0][1:, 1:]
print(f"Primal: {sdp.primal}")
T_las = extract_solution_from_X(X)
print("Ground Truth:\n", np.linalg.inv(world.T_wc))
print("SDP Solution Old:\n", T_sdp)
print("SDP Solution Las:\n", T_las)


eig_values, eig_vectors = np.linalg.eig(X)
plt.close("all")
eig_values
plt.scatter(range(len(eig_values)), eig_values)
plt.ylabel("Eigenvalues of $\mathbf{X}$ ($\lambda$)")
plt.savefig("las_eigs.png")
tikzplotlib.save("global_sdp_eigs_las.tex")

cost = projection_error(y, T_las, cam.M(), world.p_w, W)
print("SDP Solution Cost Las:", cost)
print("Local Solution Cost:", local_minima[0][0])

#%% easier way ?


