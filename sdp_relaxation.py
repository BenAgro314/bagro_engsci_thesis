from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

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

def build_measurement_constraint_matrices(num_datapoints: int, p_w: np.array) -> Tuple[List[np.array], List[np.array]]:
    As = []
    bs = []
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

    return As, bs

def build_rotation_constraint_matrices() -> Tuple[List[np.array], List[np.array]]:
    """Builds the 6 9x9 rotation matrix constraint matrices

    Returns:
        Tuple[List[np.array], List[np.array]]: First element in the tuple is a list of rotation matrix
        constraint matrices. Second element in the tuple is a list of the rhs of those constraint equations
        (b in Ax = b)
    """
    As = []
    bs = []

    A = np.zeros((9, 9))
    A[0:3, 0:3] = np.eye(3)
    As.append(A)
    bs.append(1)

    A = np.zeros((9, 9))
    A[3:6, 3:6] = np.eye(3)
    As.append(A)
    bs.append(1)

    A = np.zeros((9, 9))
    A[6:9, 6:9] = np.eye(3)
    As.append(A)
    bs.append(1)

    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)
    A = 0.5 * (A + A.T)
    As.append(A)
    bs.append(0)

    A = np.zeros((9, 9))
    A[0:3, 6:9] = np.eye(3)
    A = 0.5 * (A + A.T)
    As.append(A)
    bs.append(0)


    A = np.zeros((9, 9))
    A[3:6, 6:9] = np.eye(3)
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