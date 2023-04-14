from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.special import comb

# "getter" matrices below

def E_v(n: int, D: int) -> np.array:
    """v_n = E_{v_n} @ x

    Args:
        n (int): index of v_n
        D (int): D = x.shape[0]

    Returns:
        np.array: of shape (4, D)
    """
    E = np.zeros((4, D))
    E[0, 12 + 3*n] = 1
    E[1, 12 + 3*n + 1] = 1
    E[2, -1] = 1
    E[3, 12 + 3*n + 2] = 1
    return E

def E_omega(D: int) -> np.array:
    """omega = E_{\omega_n} @ x

    Args:
        D (int): D = x.shape[0]

    Returns:
        np.array: of shape (1, D)
    """
    E = np.zeros((1, D))
    E[0, -1] = 1
    return E

def E_Tp(p_n: np.array, D: int) -> np.array:
    """T p_{n} = E_{T_{p_n}} @ x

    Args:
        p_n (np.array): shape of (4, 1)
        D (int): D = x.shape[0]

    Returns:
        np.array: of shape (4, D)
    """
    E = np.zeros((4, D))
    E[:3, :3] = np.eye(3) * p_n[0]
    E[:3, 3:6] = np.eye(3) * p_n[1]
    E[:3, 6:9] = np.eye(3) * p_n[2]
    E[:3, 9:12] = np.eye(3) * p_n[3]
    E[-1, -1] = 1
    return E

def E_r(D: int) -> np.array:
    """r = E_r @ x

    Args:
        D (int): x.shape[0]

    Returns:
        np.array: of shape (3 , D)
    """
    E = np.zeros((3, D))
    E[:, 9:12] = np.eye(3)
    return E

def E_c(i:int , D: int) -> np.array:
    """c_i = E_{c_i} @ x

    Args:
        i (int): index in c_i
        D (int): x.shape[0]

    Returns:
        np.array: of shape (3 , D)
    """
    assert i >= 0 and i <= 2
    E = np.zeros((3, D))
    E[:, 3*i:3*i + 3] = np.eye(3, 3)
    return E

def build_homo_constraint(D: int):
    return E_omega(D).T @ E_omega(D), 1

def build_cost_matrix(D: int, y: np.array, Ws: np.array, M: np.array, r0: Optional[np.array] = None, gamma_r: float = 0, C0: Optional[np.array] = None, gamma_c: float = 0) -> np.array:
    """Build cost matrix Q for the SDP relaxation

    Args:
        num_datapoints (int): Number of landmarks in the world (N)
        y (np.array): Measurments from each landmark, (N, 4, 1)
        W (np.array): Cost weighting matrices for each datapoint, (N, 4, 4)
        M (np.array): Intrinsic camera matrix, (4, 4)
        r0 (Optional[np.array], optional): Position prior, (3, 1). Defaults to None.
        C0 (Optional[np.array], optional): Orientation prior (3, 3). Defaults to None.
        gamma_r (float, optional): Weighting on position prior cost term. Defaults to 0.
        gamma_c (float, optional): Weighting on oririention prior cost term. Defaults to 0.
    """
    Q = sum((y[n] @ E_omega(D) - M @ E_v(n, D)).T @ Ws[n] @ (y[n] @ E_omega(D) - M @ E_v(n, D)) for n in range(y.shape[0]))
    Q_r = np.zeros_like(Q)
    if r0 is not None:
        Q_r = gamma_r * (E_r(D) - r0 @ E_omega(D)).T @ (E_r(D) - r0 @ E_omega(D))
    if C0 is not None:
        pass
    Q_c = np.zeros_like(Q)
    if C0 is not None:
        Q_c[-1, :3] = - C0.T[0, :3]
        Q_c[-1, 3:6] = - C0.T[1, :3]
        Q_c[-1, 6:9] = - C0.T[2, :3]
        Q_c[-1, -1] = 3
        Q_c = gamma_c * 0.5 * (Q_c + Q_c.T)
    return Q + Q_r + Q_c

def build_measurement_constraint_matrices(D: int, p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    num_datapoints = p_w.shape[0]
    I = np.eye(4)
    _E_omega = E_omega(D)
    for n in range(num_datapoints):
        for i in [0, 1, 3]:
            _E_Tp = E_Tp(p_w[n], D)
            if i != 3:
                A =  _E_omega.T @ I[:, i:i+1].T @ _E_Tp  - E_v(n, D).T @ I[:, i:i+1] @ I[:, 2:3].T @ _E_Tp
                bs.append(0)
            else:
                A = E_v(n, D).T @ I[:, i:i+1] @ I[:, 2:3].T @ _E_Tp
                bs.append(1)
            A = 0.5 * (A + A.T)
            As.append(A)
    return As, bs


def build_rotation_constraint_matrices(D: int) -> Tuple[List[np.array], List[np.array]]:
    """Builds the 6 conventional rotation matrix constraint matrices

    Returns:
        Tuple[List[np.array], List[np.array]]: First element in the tuple is a list of rotation matrix
        constraint matrices. Second element in the tuple is a list of the rhs of those constraint equations
        (b in Ax = b)
    """
    As = []
    bs = []
    for i in range(0, 3):
        for j in range(i, 3):
            A = np.zeros((9, 9))
            A = E_c(j, D).T @ E_c(i, D)
            As.append(0.5 * (A + A.T))
            bs.append(int(i == j))

    return As, bs

def Uki(k, i, D):
    U = np.zeros((D, 3))
    eye = np.eye(3)
    if k == 0:
        U[3*i:3*i+3, 1:2] = -eye[:, 2:3]
        U[3*i:3*i+3, 2:3] = eye[:, 1:2]
    if k == 1:
        U[3*i:3*i+3, 0:1] = eye[:, 2:3]
        U[3*i:3*i+3, 2:3] = -eye[:, 0:1]
    if k == 2:
        U[3*i:3*i+3, 0:1] = -eye[:, 1:2]
        U[3*i:3*i+3, 1:2] = eye[:, 0:1]
    return U

def build_redundant_rotation_constraint_matrices(D: int) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    for i in range(0, 3):
        R_i = np.zeros((D, 3))
        R_i[i, 0] = 1
        R_i[i+3, 1] = 1
        R_i[i+6, 2] = 1
        for j in range(i, 3):
            R_j = np.zeros((D, 3))
            R_j[j, 0] = 1
            R_j[j+3, 1] = 1
            R_j[j+6, 2] = 1
            A = R_i @ R_j.T
            #A[-1, -1] = -int(i == j)
            As.append(0.5 * (A + A.T))
            #bs.append(0)
            bs.append(int(i == j))


    for i in range(0, 3):
        for k in range(0, 3):
            ek = np.zeros((3, 1))
            ek[k] = 1
            A = Uki(k, i % 3, D) @ E_c((i + 1) % 3, D) - E_omega(D).T @ ek.T @ E_c((i + 2) % 3, D)
            As.append(0.5 * (A + A.T))
            bs.append(0)

    return As, bs

def build_parallel_constraint_matrices(D: int, p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    I = np.eye(4)
    for n in range(p_w.shape[0]):
        _E_Tp = E_Tp(p_w[n], D)
        _E_v = E_v(n, D)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                A = _E_Tp.T @ I[:, j:j+1] @ I[:, i:i+1].T @ _E_v - _E_v.T @ I[:, j:j+1] @ I[:, i:i+1].T @ _E_Tp
                A = 0.5 * (A + A.T)
                As.append(A)
                bs.append(0)
    return As, bs

def E_q(i: int, j: int, D: int, N: int):
    num_qs = comb(N, 2, exact=True)
    assert i != j
    start_ind = D - num_qs - 1
    E = np.zeros((1 ,D))
    ind = sum((N - k - 1) for k in range(i)) + j - i - 1
    E[0, start_ind + ind] = 1
    return E


def build_coupling_constraint_matrices(D: int, p_w: np.array):
    N = p_w.shape[0]
    assert D > comb(N, 2, exact = True)
    As = []
    bs = []
    I = np.eye(4)
    for i in range(N):
        for j in range(i+1, N):

            # definition of q_{ij} -- (N choose 2)
            _E_q = E_q(i, j, D, N)
            A = (E_omega(D).T @ _E_q) - (E_Tp(p_w[i], D).T @  I[:, 2:3] @ I[:, 2:3].T @ E_Tp(p_w[j], D))
            As.append(0.5 * (A + A.T))
            bs.append(0)

            # q_{ij} v_i = T p_i (e_3^T T p_j) -- 3 * 2 * (N choose 2) = 12 (N choose 2)
            for k in [0, 1, 3]:
                A = E_q(i, j, D, N).T @ I[:, k:k+1].T @ E_v(i, D) - \
                    E_Tp(p_w[i], D).T @ I[:, k:k+1] @ I[:, 2:3].T @ E_Tp(p_w[j], D)
                As.append(0.5 * (A + A.T))
                bs.append(0)
                A = E_q(i, j, D, N).T @ I[:, k:k+1].T @ E_v(j, D) - \
                    E_Tp(p_w[j], D).T @ I[:, k:k+1] @ I[:, 2:3].T @ E_Tp(p_w[i], D)
                As.append(0.5 * (A + A.T))
                bs.append(0)

            # q_{ij} / q_{im} = e_3^T T p_j / (e_3^T T p_m) --- (N choose 3)
            for m in range(j + 1, N):
                A = E_Tp(p_w[m], D).T @ I[:, 2:3] @ E_q(i, j, D, N) - E_Tp(p_w[j], D).T @ I[:, 2:3] @ E_q(i, m, D, N)
                As.append(0.5 * (A + A.T))
                bs.append(0)

            # q_{ij} q_{km} = q_{im} q_{kj} = q_{jm} q_{ik} ---  3(N choose 4)
            if N >= 4:
                for m in range(j + 1, N):
                    for k in range(m + 1, N):
                        A = E_q(i, j, D, N).T @ E_q(m, k, D, N) - E_q(i, m, D, N).T @ E_q(j, k, D, N)
                        As.append(0.5 * (A + A.T))
                        bs.append(0)
                        A = E_q(i, j, D, N).T @ E_q(m, k, D, N) - E_q(j, m, D, N).T @ E_q(i, k, D, N)
                        As.append(0.5 * (A + A.T))
                        bs.append(0)
                        A = E_q(i, m, D, N).T @ E_q(j, k, D, N) - E_q(j, m, D, N).T @ E_q(i, k, D, N)
                        As.append(0.5 * (A + A.T))
                        bs.append(0)


    #assert len(As) == (4 * comb(N, 2, exact=True)) + (2 * comb(N, 3, exact=True))
    assert len(As) == (7 * comb(N, 2, exact=True)) + comb(N, 3, exact=True) + 3 * comb(N, 4, exact=True)
    assert len(bs) == len(As)
    return As, bs

def extract_solution_from_X(X: np.array) -> np.array:
    # The ordering of x needs to be c_1, c_2, c_3, r, and
    # there needs to be a homogenization variable in the last entry.
    # X = x @ x.T
    x = X[:, -1:] # last col
    C_est = x[:9].real.reshape((3, 3)).T
    r = x[9:12].real.reshape((3,1))
    v, s, ut = np.linalg.svd(C_est, full_matrices = True)
    C = v @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(ut)*np.linalg.det(v)]]) @ ut
    T = np.eye(4)
    T[:3, :3] = C
    T[:3, -1:] = r
    return T

def build_general_SDP_problem(Q: np.array, As: List[np.array], bs: List[float]):
    """Solves the SDP
    min tr(QX)
    s.t tr(A_i X) = b_i \forall i \in \{1, \dots, P}
        and X is positive semi-definite
    
    For Q, X, A all symmetric matricies

    Args:
        Q (np.array): Q as defined above. Shape = (N, N)
        As (np.array): Stores the A_i's. List of length P, each A_i is of shape (N, N)
        bs (np.array): Stores the b_i's, List of length P, each b_i a scalar
    """
    assert len(Q.shape) == 2, "Q must have two dimensions"
    n = Q.shape[0]
    assert Q.shape == (n, n), "Q must be square"
    #assert np.allclose(Q, Q.T, atol=1e-5), f"Q must be symmetric: Q - Q.T = \n{Q - Q.T}"
    p = len(As)
    assert len(bs) == p
    X = cp.Variable((n, n), PSD = True)
    constraints = [X >> 0]
    for i in range(p):
        assert As[i].shape == (n, n)
        assert np.all(As[i] == As[i].T)
        constraints.append(cp.trace(As[i] @ X) == bs[i])
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)),
                constraints)
    return prob, X

