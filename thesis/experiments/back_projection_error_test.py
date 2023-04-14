#%%
from copy import deepcopy
import os 
import cvxpy as cp
import sys

from matplotlib import pyplot as plt
from typing import Optional, Dict

from pylgmath.se3.operations import vec2tran
from pylgmath.so3.operations import hat
sys.path.append("/Users/benagro/bagro_engsci_thesis")
from thesis.relaxations.sdp_relaxation_v2 import E_Tp, build_general_SDP_problem, build_homo_constraint, build_redundant_rotation_constraint_matrices, build_rotation_constraint_matrices, extract_solution_from_X
from thesis.experiments.utils import StereoLocalizationProblem, StereoLocalizationSolution
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.iterative_sdp import iterative_sdp_solution
from thesis.simulation.sim import Camera, World, generate_random_T, generate_random_rot, generate_stereo_camera_noise, render_camera_points

from thesis.datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample, StereoLocalizationDatasetConfig
from thesis.common.utils import get_data_dir_path
from thesis.experiments.experiments_runner import run_experiment
from thesis.solvers.local_solver import stereo_localization_gauss_newton
from thesis.visualization.plotting import add_coordinate_frame, plot_percent_succ_vs_noise, plot_select_solutions_history, plot_solution_time_vs_num_landmarks
import numpy as np
import time

#%% Local Solver

def _u(y: np.array, x: np.array, M: np.array):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.
    Args:
        y (np.array): (N, 4, 1)
        x (np.array): (N, 4, 1)
        M (np.array): (4, 4)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    return y @ a.transpose((0, 2, 1)) @ x - M @ x

def _du(x: np.array, M: np.array, y: np.array):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.
    Args:
        x (np.array): (N, 4 ,1)
        M (np.array): (N, 4 ,1)
        y (np.array): (N, 4, 1), measurments
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    return  y @ a.transpose((0, 2, 1)) - M

def _odot_exp(e: np.array):
    """
    Computes e^{\odot}. See section 8.1.8 of State Estimation for Robotics (Barfoot 2022)
    Args:
        e (np.array): (N, 4, 1)
    Returns:
        np.array: (N, 4, 6)
    """
    assert len(e.shape) == 3
    eta = e[:, -1] # (N, 1)
    n = e.shape[0]
    res = np.zeros((n, 4, 6))
    res[:, :3, 3:] = -hat(e[:, :-1]) # (N, 3, 3)
    res[:, :3, :3] = eta[:, :, None] * np.eye(3) # (3, 3) # (N, 3, 3)
    return res

def _delta(x: np.array, M: np.array, y):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.

    Args:
        x (np.array): shape = (6, 1)
        M (np.array): shape = (4, 4)

    Returns:
        np.array: shape = (6, 4)
    """

    return (_du(x, M, y) @ _odot_exp(x)).transpose((0, 2, 1)) 

def _svdsolve(A: np.array, b: np.array):
    """Solve Ax = b using SVD

    Args:
        A (np.array): shape = (N, M)
        b (np.array): shape = (N, 1)

    Returns:
        x (np.array): shape = (M, 1)
    """
    u,s,v = np.linalg.svd(A)
    c = np.dot(u.T,b)
    w = np.linalg.lstsq(np.diag(s),c)[0]
    x = np.dot(v.T,w)
    return x

def back_projection_error(y: np.array, T: np.array, M: np.array, p_w: np.array, W: np.array):
    """Compute projection error

    Args:
        y (np.array): (N, 4, 1), measurments
        T (np.array): (N, 4, 4), rigid transform estimate
        M (np.array): (4, 4), camera parameters
        p_w (np.array): (N, 4, 1), homogeneous point coordinates in the world frame
        W (np.array): (N, 4, 4) or (4, 4) or scalar, weight matrix/scalar

    Returns:
        error: scalar error value
    """
    x = T @ p_w
    a = np.zeros_like(x)
    a[:, 2] = 1
    e = _u(y, x, M)
    return np.sum(e.transpose((0, 2, 1)) @ W @ e, axis = 0)[0][0]

def stereo_localization_gauss_newton_back_proj(problem: StereoLocalizationProblem, max_iters: int = 1000, min_update_norm = 1e-10, log: bool = False, num_tries: int = 1, record_history: bool = False):
    problem = deepcopy(problem)
    assert problem.y is not None and problem.T_wc is not None and problem.p_w is not None
    assert problem.W is not None, problem.M is not None
    assert problem.T_init is not None
    T_init = problem.T_init
    count_local_tries = 0
    while count_local_tries < num_tries:
        soln = _stereo_localization_gauss_newton_back_proj(
            T_init,
            problem.y,
            problem.p_w,
            problem.W,
            problem.M,
            max_iters,
            min_update_norm,
            problem.gamma_r,
            problem.r_0,
            log,
            record_history=record_history,
        )
        if soln.solved:
            break
        problem.T_init[:3, :3] = generate_random_rot()
        count_local_tries+=1
    return soln

def _stereo_localization_gauss_newton_back_proj(
    T_init: np.array,
    y: np.array,
    p_w: np.array,
    W: np.array,
    M: np.array,
    max_iters: int = 1000,
    min_update_norm: float = 1e-10,
    gamma_r: float = 0.0,
    r_0: Optional[np.array] = None,
    log: bool = True,
    record_history: bool = False,
) -> StereoLocalizationSolution:
    """Solve the stereo localization problem with a gauss-newton method

    Args:
        T_init (np.array): initial guess for transformation matrix T_cw
        y (np.array): stereo camera measurements, shape = (N, 4, 1)
        p_w (np.array): Landmark homogeneous coordinates in world frame, shape = (N, 4, 1)
        W (np.array): weight matrix/scalar shape = (N, 4, 4) or (4, 4) or scalar
        M (np.array): Stereo camera parameter matrix, shape = (4, 4)
        max_iters (int, optional): Maximum iterations before returning. Defaults to 1000.
        min_update_norm (float, optional): . Defaults to 1e-10.
        log (bool, optional): Whether or not to print loss to stdout. Defaults to True.
    """
    assert max_iters > 0, "Maximum iterations must be positive"

    i = 0
    perturb_mag = np.inf
    T_op = T_init.copy()

    if r_0 is not None:
        assert r_0.shape == (3, 1)
        r_0 = np.concatenate((r_0, np.array([[1]])), axis = 0)

    T_cw_history = [] if record_history else None
    solved = True
    while (perturb_mag > min_update_norm) and (i < max_iters):
        if record_history:
            T_cw_history.append(T_op)
        delta = _delta(T_op @ p_w, M, y)
        beta = _u(y, T_op @ p_w, M)
        A = np.sum(delta @ (W + W.T) @ delta.transpose((0, 2, 1)), axis = 0)
        b = np.sum(-delta @ (W + W.T) @ beta, axis = 0)
        if r_0 is not None: # prior on position
            b += (-gamma_r * _odot_exp(T_op[None, :, -1:]).transpose((0, 2, 1)) @ (T_op[None, :, -1:] - r_0[None, :, :])).squeeze(0)
            A += (gamma_r * _odot_exp(T_op[None, :, -1:]).transpose((0, 2, 1)) @ _odot_exp(T_op[None, :, -1:])).squeeze(0)
        epsilon = _svdsolve(A, b)
        T_op = vec2tran(epsilon) @ T_op
        if log:
            cost = back_projection_error(y, T_op, M, p_w, W)
            if r_0 is not None:
                cost += gamma_r * (T_op[:, -1:] - r_0).T @ (T_op[:, -1:] - r_0)
            print(f"Loss: {cost}")
        perturb_mag = np.linalg.norm(epsilon)
        i = i + 1
        if i == max_iters:
            solved = False
    if record_history:
        T_cw_history.append(T_op)
    cost = back_projection_error(y, T_op, M, p_w, W)
    if r_0 is not None:
        cost += gamma_r * (T_op[:, -1:] - r_0).T @ (T_op[:, -1:] - r_0)
    return StereoLocalizationSolution(solved, T_op, cost, T_cw_history)

# %% SDP

D = 13
def build_cost_matrix(y: np.array, Ws: np.array, M: np.array, p: np.array):
    I = np.eye(4)
    Q = sum((y[n] @ I[2:3, :] @ E_Tp(p[n], D) - M @ E_Tp(p[n], D)).T @ Ws[n] @ (y[n] @ I[2:3, :] @ E_Tp(p[n], D) - M @ E_Tp(p[n], D)) for n in range(y.shape[0]))
    return Q

def global_sdp_solution_back_proj(
    problem: StereoLocalizationProblem,
    return_X: bool = True,
    mosek_params: Dict[str, float] = {},
    refine: bool = True,
    record_history: bool = False,
    redundant_constraints: bool = False,
    include_coupling: bool = False,
    log: bool = False,
):
    T_cw_history = []
    num_landmarks = problem.y.shape[0]
    Ws = np.zeros((num_landmarks, 4, 4))
    for i in range(num_landmarks):
        Ws[i] = problem.W
    success = True
    Q = build_cost_matrix(problem.y, Ws, problem.M, problem.p_w)

    As, bs = build_rotation_constraint_matrices(D)

    # homogenization variable
    A, b = build_homo_constraint(D)
    As.append(A)
    bs.append(b)

    prob, X = build_general_SDP_problem(Q, As, bs)

    primal_value = None
    try:
        prob.solve(solver=cp.MOSEK, mosek_params = mosek_params)#, verbose = True)
        if prob.status != "optimal":
            assert False
        primal_value = prob.value
        if log:
            print("The optimal value from the SDP is", primal_value)
    except Exception:
        success = False
    X_sdp = X.value

    if not success:
        print(f"Global SDP failed to solve tries")

    if redundant_constraints:
        As_rot, bs_rot = build_redundant_rotation_constraint_matrices(D)
        As += As_rot
        bs += bs_rot

    if return_X:
        return X_sdp
    else:
        T = extract_solution_from_X(X_sdp)
        T_cw_history.append(T)
        if refine:
            problem = deepcopy(problem)
            problem.T_init = T
            soln = stereo_localization_gauss_newton(problem, log = False, max_iters = 100, record_history = record_history)
            if record_history:
                soln.T_cw_history = T_cw_history + soln.T_cw_history
            soln.primal_cost = primal_value
            return soln
        else:
            cost = back_projection_error(problem.y, T, problem.M, problem.p_w, problem.W)
            return StereoLocalizationSolution(True, T, cost, T_cw_history, primal_cost=primal_value)

#%% Construct problem

num_landmarks = 20

# make camera
cam = Camera(
    f_u = 484.5,
    f_v = 484.5, 
    c_u = 322,
    c_v = 247,
    b = 0.24,
    R = 0.0 * np.eye(4),
    fov_phi_range = (-np.pi/12, np.pi/12),
    fov_depth_range = (0.5, 3),
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
R = np.eye(4) * 2.0
dy = generate_stereo_camera_noise(R, size = y.shape[0])[:, :, None]
#dy[-1] = 100 * (2 * np.random.rand(4, 1) - 1)
#print(dy)
y += dy
camfig, (l_ax, r_ax) = render_camera_points(y, colors)

# %% global minima

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
    solution = stereo_localization_gauss_newton_back_proj(p_tmp, log = False)
    T_op = solution.T_cw
    local_minima = solution.cost[0][0]
    if solution.solved and local_minima < global_min_cost:
        global_min_cost = local_minima
        global_min_T = T_op


#%%
mosek_params = {}
global_sdp_soln = global_sdp_solution_back_proj(problem, return_X = False, mosek_params=mosek_params, record_history=False, refine=False, redundant_constraints=True)

#%%
print(f"Global minima: {global_min_cost}")
print(f"Global SDP primal cost: {global_sdp_soln.primal_cost}, Global SDP extracted cost: {global_sdp_soln.cost}")

#%% small experiment



gaps = []

for var in [0.1, 1.0, 2.0, 4.0]:
    for num_landmarks in [10, 20, 20, 30]:
        for _ in range(2):
            cam = Camera(
                f_u = 484.5,
                f_v = 484.5, 
                c_u = 322,
                c_v = 247,
                b = 0.24,
                R = var * np.eye(4),
                fov_phi_range = (-np.pi/12, np.pi/12),
                fov_depth_range = (0.5, 3),
            )
            world = World(
                cam = cam,
                p_wc_extent = np.array([[3], [3], [0]]),
                num_landmarks = num_landmarks,
            )
            world.clear_sim_instance()
            world.make_random_sim_instance()

            #world.make_random_sim_instance()
            #fig, ax, colors = world.render()

            # Generative camera model 
            y = cam.take_picture(world.T_wc, world.p_w)
            #camfig, (l_ax, r_ax) = render_camera_points(y, colors)

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
                solution = stereo_localization_gauss_newton_back_proj(p_tmp, log = False)
                T_op = solution.T_cw
                local_minima = solution.cost[0][0]
                if solution.solved and local_minima < global_min_cost:
                    global_min_cost = local_minima
                    global_min_T = T_op


            mosek_params = {}
            global_sdp_soln = global_sdp_solution_back_proj(problem, return_X = False, mosek_params=mosek_params, record_history=False, refine=False, redundant_constraints=True)

            print(f"Global minima: {global_min_cost}")
            print(f"Global SDP primal cost: {global_sdp_soln.primal_cost}, Global SDP extracted cost: {global_sdp_soln.cost}")
            gaps.append(global_min_cost - global_sdp_soln.primal_cost)

import tikzplotlib
 
path = "backprojection_error"
plt.hist([np.log(g) for g in gaps], bins = 20)
#plt.xlim([0, max(gaps)])
plt.xlabel("$\ln(p^{\star} - q^{\star})$")
plt.ylabel("Count")
plt.savefig(path + ".png", dpi = 400)
tikzplotlib.save(path + ".tex")