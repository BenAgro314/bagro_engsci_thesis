from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

e_1 = np.array([[1, 0, 0, 0]]).T
e_2 = np.array([[0, 1, 0, 0]]).T
e_3 = np.array([[0, 0, 1, 0]]).T
e_4 = np.array([[0, 0, 0, 1]]).T

def Bk(k: int, D: int):
    # shape = (4, D)
    Bk = np.zeros((4, D))
    Bk[0, 12 + 3*k] = 1
    Bk[1, 12 + 3*k + 1] = 1
    Bk[2, -1] = 1
    Bk[3, 12 + 3*k + 2] = 1
    return Bk

def a(D: int):
    a = np.zeros((D, 1))
    a[-1, 0] = 1
    return a

def Ck(p_k: np.array, D: int):
    # p_k.shape = (4, 1)
    Ck = np.zeros((4, D))
    Ck[:3, :3] = np.eye(3) * p_k[0]
    Ck[:3, 3:6] = np.eye(3) * p_k[1]
    Ck[:3, 6:9] = np.eye(3) * p_k[2]
    Ck[:3, 9:12] = np.eye(3) * p_k[3]
    Ck[-1, -1] = 1
    return Ck

def r_from_x_matrix(D):
    E = np.zeros((3, D))
    E[:, 9:12] = np.eye(3)
    return E

def build_homo_constraint(num_datapoints: int):
    D = 13 + 3 * num_datapoints
    A = np.zeros((D, D))
    A[-1, -1] = 1
    return A, 1

def build_cost_matrix_v2(num_datapoints: int, y: np.array, Ws: np.array, M: np.array, r0: Optional[np.array] = None, gamma_r: float = 0, C0: Optional[np.array] = None, gamma_c: float = 0) -> np.array:
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
    assert y.shape[0] == num_datapoints
    D = 13 + 3 * num_datapoints
    Q = sum((y[k] @ a(D).T - M @ Bk(k, D)).T @ Ws[k] @ (y[k] @ a(D).T - M @ Bk(k, D)) for k in range(num_datapoints))
    Q_r = np.zeros_like(Q)
    if r0 is not None:
        Q_r = gamma_r * (r_from_x_matrix(D) - r0 @ a(D).T).T @ (r_from_x_matrix(D) - r0 @ a(D).T)
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


def build_cost_matrix(num_datapoints: int, y: np.array, Ws: np.array, M: np.array, r0: Optional[np.array] = None, gamma_r: float = 0, C0: Optional[np.array] = None, gamma_c: float = 0) -> np.array:
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

    Returns:
        _type_: _description_
    """
    n = 13 + 3 * num_datapoints
    E = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
    )
    e_3 = np.array([
        [0],
        [0],
        [1],
        [0]
    ])
    Q = np.zeros((n, n))
    Q[12:-1, 12:-1] = block_diagonal(E.T @ M.T @ Ws @ M @ E, k = 0)
    g = -E.T @ M.T @ Ws @ y + E.T @ M.T @ Ws @ M @ e_3
    Q[12:-1, -1] = g.reshape(-1)
    Q[-1, 12:-1] = g.reshape(-1)
    Omega = y.transpose((0, 2, 1)) @ Ws @ y - y.transpose((0, 2, 1)) @ Ws @ M @ e_3  - e_3.T @ M.T @ Ws @ y + e_3.T @ M.T @ Ws @ M @ e_3
    Omega = Omega.sum()
    Q[-1, -1] = Omega

    Q_r_prior = np.zeros_like(Q)
    if r0 is not None:
        # prior on position
        Q_r_prior = np.zeros((n, n))
        Q_r_prior[9:12, 9:12] = np.eye(3)
        Q_r_prior[9:12, -1] = -r0.flatten()
        Q_r_prior[-1, 9:12] = -r0.flatten()
        Q_r_prior[-1, -1] = r0.T @ r0

    Q_c_prior = np.zeros_like(Q)
    if C0 is not None:
        # prior on orientation
        e_1 = np.zeros((3, 1))
        e_1[0, 0] = 1
        e_2 = np.zeros((3, 1))
        e_2[1, 0] = 1
        e_3 = np.zeros((3, 1))
        e_3[2, 0] = 1

        Q_c_prior = np.zeros((n, n))
        Q_c_prior[-1, :3] = -e_1.T @ C0.T
        Q_c_prior[-1, 3:6] = -e_2.T @ C0.T
        Q_c_prior[-1, 6:9] = -e_3.T @ C0.T
        Q_c_prior[-1, -1] = 3
        Q_c_prior = 0.5 * (Q_c_prior + Q_c_prior.T)


    Q = Q + gamma_r * Q_r_prior  + gamma_c * Q_c_prior
    return Q

def build_measurement_constraint_matrices_v2(p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    num_datapoints = p_w.shape[0]
    D = 13 + 3 * num_datapoints
    _a = a(D)
    I = np.eye(4)
    for k in range(num_datapoints):
        for i in [0, 1, 3]:
            _Ck = Ck(p_w[k], D)
            if i != 3:
                A =  _a @ I[:, i:i+1].T @ _Ck  - Bk(k, D).T @ I[:, i:i+1] @ I[:, 2:3].T @ _Ck
                bs.append(0)
            else:
                A = Bk(k, D).T @ I[:, i:i+1] @ I[:, 2:3].T @ _Ck
                bs.append(1)
            A = 0.5 * (A + A.T)
            As.append(A)
    return As, bs

def build_measurement_constraint_matrices(p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    num_datapoints = p_w.shape[0]
    n = 13 + 3 * num_datapoints
    e_1 = np.zeros((3, 1))
    e_1[0, 0] = 1
    e_2 = np.zeros((3, 1))
    e_2[1, 0] = 1
    e_3 = np.zeros((3, 1))
    e_3[2, 0] = 1
    for k in range(num_datapoints):
        A = np.zeros((n, n))
        e13 = e_1 @ e_3.T
        m = -np.expand_dims(e13, 0) * np.expand_dims(p_w[k], -1)
        m = m.transpose((1, 0, 2))
        m = m.reshape((3, -1))
        A[12 + 3*k : 12 + 3*k + 3, 0:12] = m
        A[-1, 0:12] = (e_1.T * np.expand_dims(p_w[k], -1)).flatten()
        A = 0.5 * (A + A.T)
        As.append(A)
        bs.append(0)
        
        A = np.zeros((n, n))
        e23 = e_2 @ e_3.T
        m = -np.expand_dims(e23, 0) * np.expand_dims(p_w[k], -1)
        m = m.transpose((1, 0, 2))
        m = m.reshape((3, -1))
        A[12 + 3*k : 12 + 3*k + 3, 0:12] = m
        A[-1, 0:12] = (e_2.T * np.expand_dims(p_w[k], -1)).flatten()
        A = 0.5 * (A + A.T)
        As.append(A)
        bs.append(0)
        
        A = np.zeros((n, n))
        e33 = e_3 @ e_3.T
        m = np.expand_dims(e33, 0) * np.expand_dims(p_w[k], -1)
        m = m.transpose((1, 0, 2))
        m = m.reshape((3, -1))
        A[12 + 3*k : 12 + 3*k + 3, 0:12] = m
        A = 0.5 * (A + A.T)
        As.append(A)
        bs.append(1)

    #As_other, bs_other = build_measurement_constraint_matrices_v2(p_w)
    #for A, A_other in zip(As, As_other):
    #    assert np.allclose(A, A_other), f"A:\n{A}\nA_other:\n{A_other}"
    #for b, b_other in zip(bs, bs_other):
    #    assert b == b_other, f"{b}, {b_other}"

    return As, bs

def Ei(i, dim = 9):
    E = np.zeros((3, dim))
    E[:, 3*i:3*i + 3] = np.eye(3, 3)
    return E

def build_rotation_constraint_matrices() -> Tuple[List[np.array], List[np.array]]:
    """Builds the 6 9x9 rotation matrix constraint matrices

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
            A = Ei(j).T @ Ei(i)
            As.append(0.5 * (A + A.T))
            bs.append(int(i == j))

    return As, bs

def Uki(k, i, dim = 9):
    U = np.zeros((dim, 3))
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



def build_redundant_rotation_constraint_matrices(dim) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    for i in range(0, 3):
        R_i = np.zeros((dim, 3))
        R_i[i, 0] = 1
        R_i[i+3, 1] = 1
        R_i[i+6, 2] = 1
        for j in range(i, 3):
            R_j = np.zeros((dim, 3))
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
            A = Uki(k, i % 3, dim = dim) @ Ei((i + 1) % 3, dim = dim) - a(dim) @ ek.T @ Ei((i + 2) % 3, dim = dim)
            As.append(0.5 * (A + A.T))
            bs.append(0)

    return As, bs

def build_parallel_constraint_matrices(p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
    num_datapoints = p_w.shape[0]
    D = 13 + 3 * num_datapoints
    I = np.eye(4)
    for k in range(num_datapoints):
        C_k = Ck(p_w[k], D)
        B_k = Bk(k, D)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                A = C_k.T @ I[:, j:j+1] @ I[:, i:i+1].T @ B_k - B_k.T @ I[:, j:j+1] @ I[:, i:i+1].T @ C_k
                A = 0.5 * (A + A.T)
                As.append(A)
                bs.append(0)
    return As, bs


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

def block_diagonal(x, k):
    ''' x should be a tensor-3 (#num matrices, n,n)
        k : int
        Diagonal in question. it is 0 in case of main diagonal. 
        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
    '''

    shape = x.shape
    n = shape[-1]

    absk = abs(k)

    indx = np.repeat(np.arange(n),n)
    indy = np.tile(np.arange(n),n)

    indx = np.concatenate([indx + a * n for a in range(shape[0])])
    indy = np.concatenate([indy + a * n for a in range(shape[0])])

    if k<0: 
        indx += n*absk
    else:
        indy += n*absk

    block = np.zeros(((shape[0]+absk)*n,(shape[0]+absk)*n))
    block[(indx,indy)] = x.flatten()

    return block

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