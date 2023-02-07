
import numpy as np
import cvxpy as cp
import sys
import scipy.io
import matplotlib.pyplot as plt
sys.path.append("/home/agrobenj/bagro_engsci_thesis")
sys.path.append("/home/agrobenj/bagro_engsci_thesis/thesis/")


from thesis.ncpol2sdpa import generate_variables, SdpRelaxation
from relaxations.sdp_relaxation import build_general_SDP_problem


def local_solver(a, y, x_init, num_iters, eps = 1e-5):
    x_op = x_init
    for i in range(num_iters):
        u = y - (1/(x_op - a)) 
        du = 1/((x_op - a)**2)
        dx = -np.sum(u * du) / np.sum(du * du)
        x_op = x_op + dx
        if np.abs(dx) < eps:
            return x_op
    return None


#%% generate ground truth

x = 5
N = 10
# ground truth landmark positions
a = x + 1 + np.random.rand(N, 1) * 10 #np.linspace(0, N, N).reshape((N, 1)) 
# measurments

sigma = 0.1
y = (1 / (x - a)) + (sigma * np.random.randn(N, 1))

#%% Construct SDP

def e_x():
    e = np.zeros((N + 2, 1))
    e[0] = 1
    return e

def e_z(j):
    e = np.zeros((N + 2, 1))
    e[1 + j] = 1
    return e

def e_omega():
    e = np.zeros((N + 2, 1))
    e[-1] = 1
    return e

# cost
Q = sum([(y[j] * e_omega().T - e_z(j).T).T @ (y[j] * e_omega().T - e_z(j).T) for j in range(N)])

# measurment constraints
As = []
bs = []
for j in range(N):
    A = e_z(j) @ e_x().T - a[j] * e_z(j) @ e_omega().T 
    As.append(0.5 * (A + A.T))
    bs.append(1)

# homogenization constraints
As.append(e_omega() @ e_omega().T)
bs.append(1)

As_redun = []
bs_redun = []

# redundant constraints
for i in range(N):
    for j in range(i+1, N):
        A = (a[j] - a[i]) * e_z(j) @ e_z(i).T + e_z(i) @ e_omega().T - e_z(j) @ e_omega().T
        As_redun.append(0.5 * (A + A.T))
        bs_redun.append(0)

mdict = {
    "Q": Q,
    "As": As,
    "bs": bs,
    "As_redun": As_redun,
    "bs_redun": bs_redun,
    "a": a,
    "x": x,
    "y": y,
}

scipy.io.savemat(f"/home/agrobenj/bagro_engsci_thesis/thesis/matlab/1D_problem_{N}_landmarks_full.mat", mdict)

#%% load problem

mdict = scipy.io.loadmat(f"/home/agrobenj/bagro_engsci_thesis/thesis/matlab/1D_problem_10_landmarks_full.mat")
Q = mdict["Q"]
As = mdict["As"]
bs = mdict["bs"][0]
As_redun = mdict["As_redun"]
bs_redun = mdict["bs_redun"][0]
x = mdict["x"].item()
y = mdict["y"]
a = mdict["a"]

#%%

# local solver

local_solutions = []
for x_init in np.linspace(-10, 10, 1000):
    x_local = local_solver(a = a, y = y, x_init = x_init, num_iters=100)
    if x_local is not None:
        local_solutions.append(x_local)
        #print(f"x local: {x_local}")
local_solutions = np.array(local_solutions)
costs = np.sum((y - (1 / (local_solutions - a))) **2, axis = 0) 
#cost = 
#global_solution = 
#worst_solution 
#print(global_solution)
min_local_ind = np.argmin(costs)
min_local_cost = costs[min_local_ind]
best_local_solution = local_solutions[min_local_ind]
#x_local = np.array([best_local_solution] + list((1 / (best_local_solution - a)).flatten()) + [1]).reshape((-1, 1))
#X_local = x_local @ x_local.T


#%% SDP solution (MOSEK)


use_redun = True

As_ineq = []
for i in range(N):
    A = a[i]  * e_omega() @ e_omega().T - e_omega() @ e_x().T
    As_ineq.append(0.5 * (A + A.T))

if use_redun:
    prob, X = build_general_SDP_problem(
        Q,
        np.concatenate((As, As_redun), axis = 0),
        np.concatenate((bs, bs_redun), axis = 0),
    )
else:
    prob, X = build_general_SDP_problem(
        Q,
        As,
        bs,
    )

#n = Q.shape[0]
#X = cp.Variable((n, n), PSD = True)
#constraints = [X >> 0]
#for i in range(len(As)):
#    constraints.append(cp.trace(As[i] @ X) == bs[i])
#if use_redun:
#    for i in range(len(As_redun)):
#        constraints.append(cp.trace(As_redun[i] @ X) == bs_redun[i])
#for i in range(len(As_ineq)):
#    constraints.append(cp.trace(As_ineq[i] @ X) >= 0)

#prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)),
#            constraints)

prob.solve(solver=cp.MOSEK)
print(f"Primal: {prob.value}")

X_sdp = X.value

x_sdp = X_sdp[0, -1] # approx
print(f"SDP soln: {x_sdp}, cost: {np.sum((y - (1 / (x_sdp - a))) ** 2)}")
print(f"Best local soln: {best_local_solution}, cost: {min_local_cost}")

U, S, V = np.linalg.svd(X_sdp, hermitian=True)
print(f"eig gap: {np.log10(S[0]) - np.log10(S[1])}")
plt.plot(S)
plt.yscale("log")
plt.show()

#%% Lasseres

use_redun = False

n_vars = Q.shape[0]
level = 2 
x = generate_variables('x', n_vars)
obj = np.dot(x, np.dot(Q, np.transpose(x)))


if use_redun:
    equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b  for A, b in zip(np.concatenate((As, As_redun), axis = 0), np.concatenate((bs, bs_redun), axis = 0))]
else:
    equalities = [np.dot(x, np.dot(A, np.transpose(x))) - b  for A, b in zip(As, bs)]

#inequalities = [-x[1]**2 + x[1] + 0.5>=0]
sdp = SdpRelaxation(x)


print(f"Build optimization problem!")
sdp.get_relaxation(level, objective=obj, equalities=equalities)
print(f"Relaxed problem!")
print(f"Solving problem")
sdp.solve(solver = 'mosek')
print(f"Done solving problem")
X_lass = sdp.x_mat[0][1:, 1:]
x_lass = X_lass[0, Q.shape[0] - 1] # approx

print(f"Primal: {sdp.primal}")
print(f"SDP lass soln: {x_sdp}, cost: {np.sum((y - (1 / (x_lass - a)))**2)}")
print(f"Best local soln: {best_local_solution}, cost: {min_local_cost}")

U, S, V = np.linalg.svd(X_sdp, hermitian=True)
print(f"eig gap: {np.log10(S[0]) - np.log10(S[1])}")
plt.plot(S)
plt.yscale("log")
plt.show()
