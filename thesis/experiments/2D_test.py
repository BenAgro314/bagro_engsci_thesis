#%%
import numpy as np
import sys
from ncpol2sdpa import *
import cvxpy as cp
sys.path.append("/Users/benagro/bagro_engsci_thesis")
from thesis.relaxations.sdp_relaxation_v2 import build_general_SDP_problem


def forward_exact(T, p_w):
    assert T.shape == (3, 3)
    assert p_w.shape[1:] == (3, 1) and len(p_w.shape) == 3
    assert np.all(p_w[:, -1, 0] == 1)
    e = np.eye(3)
    y = e[:, 0:1].T @ (T @ p_w) / (e[:, 1:2].T @ T @ p_w)
    return y.reshape(-1)

def forward_noisy(T, p_w, sigma):
    y = forward_exact(T, p_w) + (sigma * np.random.randn())
    return y

def generate_problem(N, sigma):
    p_w = np.concatenate((np.random.rand(N, 2, 1), np.ones((N, 1, 1))), axis = 1)
    theta = 2 * np.pi * np.random.rand()
    T = np.linalg.inv(np.array(
        [
            [np.cos(theta), -np.sin(theta), np.random.randn()],
            [np.sin(theta), np.cos(theta), np.random.randn()],
            [0, 0, 1]
        ]
    ))
    y = forward_noisy(T, p_w, sigma)
    W = np.ones((N,))
    return y, p_w, T, W

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
    assert len(W) == N and len(W.shape) == 1
    assert len(y) == N and len(y.shape) == 1

    dim = 2 * N + 7

    e = np.eye(3)

    As = [] 
    bs = []

    _E_omega = E_omega(dim)
    # Cost
    Q = sum(
        (y[n] * _E_omega - e[:, 0:1].T @ E_v(n, dim)).T * W[n] * (y[n] * _E_omega - e[:, 0:1].T @ E_v(n, dim))
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

#%%

N = 3
sigma = 0.05
y, p_w, T, W = generate_problem(N, sigma)
Q, As, bs = build_SDP(p_w, y, W)
prob, X = build_general_SDP_problem(Q, As, bs)
prob.solve(solver=cp.MOSEK, mosek_params = {}, verbose = True)
X_value = X.value

#%%


n_vars = Q.shape[0]
level = 2
x = generate_variables('x', n_vars)
obj = np.dot(x, np.dot(Q, np.transpose(x)))
equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b for A, b in zip(As, bs)]
sdp = SdpRelaxation(x)
print(f"Build optimization problem!")
sdp.get_relaxation(level, objective=obj, equalities=equalities)
print(f"Relaxed problem!")
#%%
print(f"Solving problem")
sdp.solve(solver = 'mosek')
X = sdp.x_mat[0][1:, 1:]
