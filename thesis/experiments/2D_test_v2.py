#%%
import itertools
import numpy as np
import sys
import matplotlib.pyplot as plt
import cvxpy as cp
sys.path.append("/Users/benagro/bagro_engsci_thesis")
from thesis.relaxations.sdp_relaxation_v2 import build_general_SDP_problem
from thesis.ncpol2sdpa import *

#M = np.array([1, 0, 0]).reshape((1, 3))
M = np.array([[1, 0, 1], [1, 0, -1]]).reshape((2, 3))

def forward_exact(T, p_w):
    assert T.shape == (3, 3)
    assert p_w.shape[1:] == (3, 1) and len(p_w.shape) == 3
    assert np.all(p_w[:, -1, 0] == 1)
    e = np.eye(3)
    y = M @ (T @ p_w) / (e[:, 1:2].T @ T @ p_w)
    return y.reshape(-1, M.shape[0], 1)

def forward_noisy(T, p_w, sigma):
    y = forward_exact(T, p_w) + (sigma * np.random.randn())
    return y

def _T(phi):
    # phi: 3, 1
    x = phi[0, 0]
    y = phi[1, 0]
    theta = phi[2, 0]
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ]
    )



def generate_problem(N, sigma):

    phi = np.array([np.random.rand(), np.random.rand(), 2 * np.pi * np.random.rand()]).reshape((3, -1))
    T = _T(phi)


    p_s = np.concatenate((2 * np.random.rand(N, 1, 1) - 1.0, (2.0 * np.random.rand(N, 1, 1)) + 1.0, np.ones((N, 1, 1))), axis = 1)
    p_w = np.linalg.inv(T) @ p_s

    y = forward_noisy(T, p_w, sigma)
    W = np.stack([np.eye((M.shape[0]))] * N)
    return y, p_w, phi, W

# x = [c_1 ; c_2 ; r ; u_1 ; u_2 ; \dots ; u_N ; \omega], 2N + 7 variables

def E_c(i, dim):
    assert i in [0, 1]
    e = np.zeros((2, dim))
    e[:, 2*i:2*i+2] = np.eye(2)
    return e

def E_r(dim):
    e = np.zeros((2, dim))
    e[:, 4:6] = np.eye(2)
    return e

def E_v(i, dim):
    e = np.zeros((3, dim))
    e[0, 6 + 2*i] = 1
    e[1, -1] = 1
    e[-1, 6 + 2*i + 1] = 1
    return e

def E_omega(dim):
    e = np.zeros((1, dim))
    e[0, -1] = 1
    return e

def E_Tp(p_n, dim):
    e = np.zeros((3, dim))
    e[:2, :2] = np.eye(2) * p_n[0]
    e[:2, 2:4] = np.eye(2) * p_n[1]
    e[:2, 4:6] = np.eye(2) * p_n[2]
    e[-1, -1] = 1
    return e

def build_SDP(p_w, y, W):
    assert len(p_w.shape) == 3 and p_w.shape[1:] == (3, 1) and np.all(p_w[:, -1]) == 1
    N = p_w.shape[0]
    #assert len(W) == N and len(W.shape) == 1
    #assert len(y) == N and len(y.shape) == 1

    dim = 2 * N + 7

    e = np.eye(3)

    As = [] 
    bs = []

    _E_omega = E_omega(dim)
    # Cost
    Q = sum(
        (y[n] @ _E_omega - M @ E_v(n, dim)).T @ W[n] @ (y[n] @ _E_omega - M @ E_v(n, dim))
        for n in range(N)
    )

    # homogenization constraint
    As.append(_E_omega.T @ _E_omega)
    bs.append(1)

    # Rotation Matrix constraints
    for i in range(2):
        for j in range(i, 2):
            A = E_c(j, dim).T @ E_c(i, dim)
            As.append(0.5 * (A + A.T))
            bs.append(int(i == j))

    # Measurment constraints
    for n in range(N):
        for k in [0, 2]:
            A = E_v(n, dim).T @ e[:, k:k+1] @ e[:, 1:2].T @ E_Tp(p_w[n], dim) - \
                _E_omega.T @ e[:, k:k+1].T @ E_Tp(p_w[n], dim)
            As.append(0.5 * (A + A.T))
            bs.append(0)

    return Q, As, bs


def build_x(T, p_w):
    x = [T[:-1, 0:1], T[:-1, 1:2], T[:-1, 2:3]]
    e = np.eye(3)
    v = T @ p_w / (e[:, 1:2].T @ T @ p_w)
    u = v[:, [0, -1]]
    for un in u:
        x.append(un)
    x.append(np.array([1])[:, None])
    x = np.concatenate(x, axis = 0)
    return x

# --- local solver functions -----

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

def _qn(phi, p_w):
    # p_w.shape = (N, 3, 1)
    T = _T(phi) # (3, 3)
    return T @ p_w # (N, 3, 1)

def _un(y, p_w, phi):
    # y = (N, 1)
    y = y.reshape((p_w.shape[0], M.shape[0], 1))
    q = _qn(phi, p_w) # (N, 3, 1)
    I = np.eye(3) # (1, 3) -> (N, 3, 3)
    u = y - ((M @ q) / (I[:, 1:2].T @ q)) # (N, 1, 1)
    #assert u.shape == (p_w.shape[0], 1, 1)
    return u
    
def _dq_dphi(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    theta = phi[2, 0]
    res = np.zeros((p_w.shape[0], 3, 3))
    res[:, 0, 0] = 1
    res[:, 1, 1] = 1
    res[:, 0, 2] = -np.sin(theta) * p_w[:, 0, 0] - np.cos(theta) * p_w[:, 1, 0]
    res[:, 1, 2] = np.cos(theta) * p_w[:, 0, 0] - np.sin(theta) * p_w[:, 1, 0]
    return res # (N, 3, 3)

def _du_dq(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    q = _qn(phi, p_w) # (N, 3, 1)
    I = np.eye(3)
    du_dq =  (1 / (I[:, 1:2].T @ q)) * ((1 / (I[:, 1:2].T @ q)) * (M @ q @ I[:, 1:2].T) - M) # (N, 1, 3)
    #assert du_dq.shape == (p_w.shape[0], 1, 3)
    return du_dq

def _du_dphi(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    du_dphi = _du_dq(p_w, phi) @ _dq_dphi(p_w, phi) # (N, 1, 3)
    # assert du_dphi.shape == (p_w.shape[0], 1, 3)
    return du_dphi
    

def _cost(p_w, W, y, phi):
    # W: (N, 1, 1)
    u = _un(y, p_w, phi)
    return np.sum(u.transpose((0, 2, 1)) @ W @ u, axis = 0)

def local_solver(p_w, y, W, init_phi, max_iters = 100, min_update_norm=1e-10, log = False):
    # see notes for math
    # W: (N)
    W = W.reshape((W.shape[0], M.shape[0] , M.shape[0])) # (N, 2, 2)
    phi = init_phi
    i = 0
    perturb_mag = np.inf
    solved = True
    while (perturb_mag > min_update_norm) and (i < max_iters):
        if log:
            print(f"Current cost: {_cost(p_w, W, y, phi)}")
        du_dphi = _du_dphi(p_w, phi) # (N, 1, 3)
        # (3, 1)
        u = _un(y, p_w, phi) # (N, M.shape[0], 1)
        b = - np.sum(u.transpose((0, 2, 1)) @ W @ du_dphi, axis = 0)  # (1, 3)
        A = np.sum(du_dphi.transpose((0, 2, 1)) @ W @ du_dphi, axis = 0) # (3, 3)
        dphi = _svdsolve(A.T, b.T) # (3 , 1)
        phi += dphi
        perturb_mag = np.linalg.norm(dphi)
        i += 1
        if i == max_iters:
            solved = False
    cost = _cost(p_w, W, y, phi)
    return solved, phi, cost

# ---- plotting -----

def plot_soln(p_w, phi, camera_color = 'k', ax = None, name = ""):
    if ax is None:
        _, ax = plt.subplots()
    y = forward_exact(_T(phi), p_w) # (N, )

    #y = np.linspace(-1, 1, 10)

    #plane_pt_s = np.stack([y, np.ones_like(y), np.ones_like(y)], axis = -1)[:, :, None]
    #plane_pt_w = (np.linalg.inv(_T(phi)) @ plane_pt_s)[:, :-1, :] # (N, 2, 1)

    #ax.scatter(plane_pt_w[:, 0], plane_pt_w[:, 1], color = camera_color)


    ax.scatter(p_w[:, 0], p_w[:, 1], color = 'b')

    #to_x = phi[0, 0] + 0.1 * np.cos(phi[2, 0])
    #to_y = phi[1, 0] + 0.1 * np.sin(phi[2, 0])
    l = 1
    looking_towards_s = np.array(
        [
            [
                [0],
                [0],
                [1],
            ],
            [
                [0],
                [l],
                [1],
            ]
        ]
    )
    looking_towards_w = np.linalg.inv(_T(phi)) @ looking_towards_s

    plane_s = np.array(
        [
            [
                [-1],
                [1],
                [1]
            ],
            [
                [1],
                [1],
                [1]
            ],
        ]
    )

    plane_w = np.linalg.inv(_T(phi)) @ plane_s

    ax.set_aspect('equal', adjustable='box')
    #ax.plot([phi[0, 0], to_x], [phi[1, 0], to_y], color = 'k') 
    ax.plot(looking_towards_w[:, 0], looking_towards_w[:, 1], color = camera_color, label = name) 
    ax.plot(plane_w[:, 0], plane_w[:, 1], alpha = 0.5, color = camera_color) 
    ax.scatter(looking_towards_w[0, 0], looking_towards_w[0, 1], color = camera_color)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.legend()
    return ax


#%% generate problem

N = 3
sigma = 0.05
y, p_w, phi_gt, W = generate_problem(N, sigma)

#%% local solver

global_min = np.inf
global_soln = None

for i in range(20):
    init_phi = np.array([np.random.rand(), np.random.rand(), 2 * np.pi * np.random.rand()]).reshape((3, -1))
    solved, phi_local, local_cost = local_solver(p_w, y, W, init_phi = init_phi)
    if solved and (local_cost < global_min):
        global_soln = phi_local
        global_min = local_cost

print(f"Ground truth:\n{phi_gt}")
print(f"Global soln:\n{global_soln}")


# make x from best local solution
T_global = _T(global_soln)
x_global = T_global[:2, :].T.reshape(-1, 1)
I = np.eye(3)
u = (T_global @ p_w) / (I[:, 1:2].T @ T_global @ p_w)
u = u[:, [0, 2], :].reshape(-1, 1)
x_global = np.concatenate((x_global, u, [[1, ]]), axis = 0)
X_global = x_global @ x_global.T


Q, As, bs = build_SDP(p_w, y, W)

assert np.isclose(np.trace(Q @ X_global), global_min)

for A, b in zip(As, bs):
    assert np.isclose(np.trace(A @ X_global), b)

#%%

prob, X = build_general_SDP_problem(Q, As, bs)
prob.solve(solver=cp.MOSEK, mosek_params = {}, verbose = False)
X_value = X.value


def extract_solution(X):
    C = X[:4, -1:]
    C = np.concatenate([C[:2, -1:], np.zeros((1, 1)), C[2:, -1:], np.zeros((3, 1)), np.ones((1, 1))], axis=0)
    C = C.reshape((3, 3)).T
    r = X[4:6, -1:]
    v, s, ut = np.linalg.svd(C, full_matrices = True)
    C = v @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(ut)*np.linalg.det(v)]]) @ ut
    T_est = np.eye(3)
    T_est[:2, :2] = C[:2, :2]
    T_est[:2, -1:] = r
    return T_est

def phi_from_T(T):
    theta_est = np.arctan2(T[1, 0], T[0, 0]) 
    phi_est = np.array([T[0, -1], T[1, -1], theta_est])[:, None]
    return phi_est

T_est = extract_solution(X_value)
phi_est = phi_from_T(T_est)


def extract_solution_lag(sdp):

    monomials = sdp.monomial_sets[0]
    inds_of_interest = [
        monomials.index(sdp.variables[i] * sdp.variables[-1])
        for i in range(0, 6)
    ]
    x = np.array([sdp.x_mat[0][0, ind] for ind in inds_of_interest])
    C = x[:4].reshape(4, 1)
    C = np.concatenate([C[:2, -1:], np.zeros((1, 1)), C[2:, -1:], np.zeros((3, 1)), np.ones((1, 1))], axis=0)
    C = C.reshape((3, 3)).T
    r = x[4:6].reshape((2, 1))
    v, s, ut = np.linalg.svd(C, full_matrices = True)
    C = v @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(ut)*np.linalg.det(v)]]) @ ut
    T_est = np.eye(3)
    T_est[:2, :2] = C[:2, :2]
    T_est[:2, -1:] = r
    return T_est

n_vars = Q.shape[0]
level = 2
x = generate_variables('x', n_vars)
obj = np.dot(x, np.dot(Q, np.transpose(x)))
equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b for A, b in zip(As, bs)]
sdp = SdpRelaxation(x)
print(f"Build optimization problem!")
sdp.get_relaxation(level, objective=obj, equalities=equalities)
print(f"Relaxed problem!")
print(f"Solving problem")
sdp.solve(solver = 'mosek')
print(f"Solved Problem")

T_lag = extract_solution_lag(sdp)
phi_lag = phi_from_T(T_lag)
cost = _cost(p_w, W, y, phi_lag)

#%%

print(f"Global min cost: {global_min[0][0]}")
print(f"Global SDP Primal: {prob.value}")
print(f"Lass Primal: {sdp.primal}")
ax = plot_soln(p_w, phi_gt, camera_color = 'k', name = 'gt')
plot_soln(p_w, global_soln, camera_color = 'orange', name='global minima', ax = ax)
plot_soln(p_w, phi_est, camera_color = 'm', ax = ax, name = 'global SDP')
plot_soln(p_w, phi_lag, camera_color = 'green', ax = ax, name = "lasserre's")

#%% sparse lasserre's


n_vars = Q.shape[0]
x = generate_variables('x', n_vars)
obj = np.dot(x, np.dot(Q, np.transpose(x)))
equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b for A, b in zip(As, bs)]
sparse_sdp = SdpRelaxation(x)
#extramonomials = [x[0] * x[1]]
extramonomials = [] #x 
#for p in itertools.product(x, x):
    #extramonomials.append(p[0] * p[1])
sparse_sdp.get_relaxation(level = 2, objective=obj, equalities=equalities, extramonomials = extramonomials)
sparse_sdp.solve(solver = 'mosek')
print(f"Sparse Lass Primal: {sparse_sdp.primal}")
print(f"Is tight: {sparse_sdp.primal >= global_min[0][0]}")

#%%
print(f"Global min cost: {global_min[0][0]}")
print(f"Global SDP Primal: {prob.value}")
print(f"Lass Primal: {sdp.primal}")
ax = plot_soln(p_w, phi_gt, camera_color = 'k', name = 'gt')
plot_soln(p_w, global_soln, camera_color = 'orange', name='global minima', ax = ax)
plot_soln(p_w, phi_est, camera_color = 'm', ax = ax, name = 'global SDP')
plot_soln(p_w, phi_lag, camera_color = 'green', ax = ax, name = "lasserre's")