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