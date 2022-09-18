from typing import List

import cvxpy as cp
import numpy as np


def build_SDP_problem(Q: np.array, As: List[np.array], bs: List[float]):
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
    assert np.all(Q == Q.T), "Q must be symmetric"
    p = len(As)
    assert len(bs) == p
    X = cp.Variable((n, n), symmetric = True)
    constraints = [X >> 0]
    for i in range(p):
        assert As[i].shape == (n, n)
        assert np.all(As[i] == As[i].T)
        constraints.append(cp.trace(As[i] @ X) == bs[i])
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)),
                constraints)
    return prob