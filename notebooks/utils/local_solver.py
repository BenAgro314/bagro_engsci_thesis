import numpy as np
from pylgmath.so3.operations import hat
from . import sim

def u(y: np.array, x: np.array, M: np.array):
    """
    Args:
        y (np.array): (N, 4, 1)
        x (np.array): (N, 4, 1)
        M (np.array): (4, 4)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    return y - 1/(a.transpose((0, 2, 1)) @ x) * (M @ x)

def du(x: np.array, M: np.array):
    """
    Args:
        x (np.array): (N, 4 ,1)
        M (np.array): (N, 4 ,1)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    a_times_x = a.transpose((0, 2, 1)) @ x
    return ((1/a_times_x)**2) * M @ x @ a.transpose((0, 2, 1)) - (1/a_times_x) * M

def odot_exp(e: np.array):
    """
    Args:
        e (np.array): (N, 4, 1)
    """
    assert len(e.shape) == 3
    eta = e[:, -1] # (N, 1)
    n = e.shape[0]
    res = np.zeros((n, 4, 6))
    res[:, :3, 3:] = -hat(e[:, :-1]) # (N, 3, 3)
    res[:, :3, :3] = eta[:, :, None] * np.eye(3) # (3, 3) # (N, 3, 3)
    return res

def delta(x, M):
    return (du(x, M) @ odot_exp(x)).transpose((0, 2, 1)) 

def svdsolve(A: np.array, b: np.array):
    u,s,v = np.linalg.svd(A)
    c = np.dot(u.T,b)
    w = np.linalg.solve(np.diag(s),c)
    x = np.dot(v.T,w)
    return x

def loss(y: np.array, T: np.array, M: np.array, p_w: np.array, W: np.array):
    y_pred = sim.generative_camera_model(M, T, p_w)
    e = y - y_pred
    return np.sum(e.transpose((0, 2, 1)) @ W @ e, axis = 0)[0][0]