import numpy as np
from pylgmath.so3.operations import hat
from pylgmath.se3.operations import vec2tran
import sim

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
    return y - 1/(a.transpose((0, 2, 1)) @ x) * (M @ x)

def _du(x: np.array, M: np.array):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.
    Args:
        x (np.array): (N, 4 ,1)
        M (np.array): (N, 4 ,1)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    a_times_x = a.transpose((0, 2, 1)) @ x
    return ((1/a_times_x)**2) * M @ x @ a.transpose((0, 2, 1)) - (1/a_times_x) * M

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

def _delta(x: np.array, M: np.array):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.

    Args:
        x (np.array): shape = (6, 1)
        M (np.array): shape = (4, 4)

    Returns:
        np.array: shape = (6, 4)
    """

    return (_du(x, M) @ _odot_exp(x)).transpose((0, 2, 1)) 

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
    w = np.linalg.solve(np.diag(s),c)
    x = np.dot(v.T,w)
    return x

def projection_error(y: np.array, T: np.array, M: np.array, p_w: np.array, W: np.array):
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
    y_pred = sim.generative_camera_model(M, T, p_w)
    e = y - y_pred
    return np.sum(e.transpose((0, 2, 1)) @ W @ e, axis = 0)[0][0]

def stereo_localization_gauss_newton(T_init: np.array, y: np.array, p_w: np.array, W: np.array, M: np.array, max_iters: int = 1000, min_update_norm: float = 1e-10, log: bool = True):
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

    while (perturb_mag > min_update_norm) and (i < max_iters):
        delta = _delta(T_op @ p_w, M)
        beta = _u(y, T_op @ p_w, M)
        A = np.sum(delta @ (W + W.T) @ delta.transpose((0, 2, 1)), axis = 0)
        b = np.sum(-delta @ (W + W.T) @ beta, axis = 0)
        epsilon = _svdsolve(A, b)
        T_op = vec2tran(epsilon) @ T_op
        if log:
            print(f"Loss: {projection_error(y, T_op, M, p_w, W)}")
        perturb_mag = np.linalg.norm(epsilon)
        i = i + 1
    return T_op
