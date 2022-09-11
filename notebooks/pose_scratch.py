#%% imports 
import numpy as np
import pylgmath
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from pylgmath.so3.operations import vec2rot

from utils import plotting

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plotting.add_coordinate_frame(np.eye(4), ax, "$\mathfrak{F}_w$")

a1 = np.array([[1], [0], [0]])
a2 = np.array([[0], [0], [1]])
a3 = np.array([[1], [0], [0]])

C_wc = vec2rot(0 * a3/np.linalg.norm(a3)) @ vec2rot(3*np.pi/4 * a2/np.linalg.norm(a2)) @ vec2rot(-np.pi/2 * a1/np.linalg.norm(a1)) 
T_wc = np.eye(4)
T_wc[:3, :3] = C_wc
T_wc[:-1, -1] = [3, 3, 0]

plotting.add_coordinate_frame(T_wc, ax, "$\mathfrak{F}_c$")

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

# add points

N = 5
p_c = np.random.rand(N, 3) * np.array([1, 1, 3]) + np.array([-0.5, -0.5, 1]) # points in camera frame
homo_p_c = np.concatenate((p_c, np.ones_like(p_c[:, 0:1])), axis = 1)
p_w = (T_wc @ homo_p_c[:, :, None]).squeeze(-1)
colors = {}
cmap = plt.cm.get_cmap("hsv", N)
for i, p in enumerate(p_w):
    colors[i] = cmap(i) 
    ax.scatter3D(p[0], p[1], p[2], color = colors[i])

fig.savefig("figs/axes.png")

for ii in range(0,360,10):
    ax.view_init(elev=10., azim=ii)
    fig.savefig("figs/movie%d.png" % ii)

#%% projection visualization

# camera parameters
f_u = 100 # focal length in horizonal pixels
f_v = 100 # focal length in vertical pixels
c_u = 50 # pinhole projection in horizonal pixels
c_v = 50 # pinhold projection in vertical pixels
b = 0.2 # baseline (meters)

M = np.array(
    [
        [f_u, 0, c_u, f_u * b / 2],
        [0, f_v, c_v, 0],
        [f_u, 0, c_u, -f_u * b / 2],
        [0, f_v, c_v, 0],
    ]
)

def camera_model(M, T_cw, homo_p_w):
    p_c = T_cw @ homo_p_w
    assert np.all(p_c[:, 2] > 0)
    return M @ p_c / p_c[:, None, 2]

#homo_points = np.concatenate((points, np.ones_like(points[:, 0:1])), axis = 1)[:, :, None]

T_cw = np.linalg.inv(T_wc)
camera_points = camera_model(M, T_cw, p_w[:, :, None]).squeeze(-1)
left = camera_points[:, :2]
right = camera_points[:, 2:]

camfig, (lax, rax) = plt.subplots(1, 2, sharex=True, sharey=True)

lax.invert_yaxis()

lax.set_title("Left Image")
rax.set_title("Right Image")
for i, (pl, pr) in enumerate(zip(left, right)):
    lax.scatter(pl[0], pl[1], color = colors[i])
    rax.scatter(pr[0], pr[1], color = colors[i])

rax.set_aspect('equal')
lax.set_aspect('equal')

camfig.savefig("figs/camera.png")

#%% local solver

p_w = p_w[:, :, None] # (N, 4, 1)
y = camera_points[:, :, None] # (N, 4, 1)

#%%
def dot_exp(e):
    assert len(e.shape) == 3
    N = e.shape[0]
    assert e.shape[-2:] == (4, 1)
    res = np.zeros((N, 4, 6))
    eta = e[:, -1][:, :, None]
    epsilon = e[:, :-1]
    res[:, :3, :3] = eta * np.eye(3)[None, :, :]
    res[:, :3, 3:] = -1 * pylgmath.so3.operations.hat(epsilon)
    return res


def u(x: np.array):
    """
    Args:
        x (np.array): (N, 4, 1)
    """
    a = np.zeros(x.shape)
    a[:, 2] = 1
    ax = a.transpose((0, 2, 1)) @ x
    return y - (M @ x) / ax # (N, 4, 1)

def du(x: np.array):
    """
    Args:
        x (np.array): (N, 4, 1)
    """
    a = np.zeros(x.shape)
    a[:, 2] = 1
    ax = a.transpose((0, 2, 1)) @ x

    return (1/ax) * M - (1/ax)**2 * M @ x @ a.transpose((0, 2, 1))

T_op = np.eye(4) # (4, 4), TODO: how to initialize this?

for i in range(10):

    beta = u(T_op @ p_w)
    assert len(beta.shape) == 3
    assert beta.shape[-2:] == (4, 1)
    delta = (du(T_op @ p_w) @ dot_exp(T_op @ p_w)).transpose((0, 2, 1))
    assert len(delta.shape) == 3
    assert delta.shape[-2:] == (6, 4)

    b = -np.sum(delta @ beta, axis = 0) # (6, 1)
    A = np.sum(delta @ delta.transpose((0, 2, 1)), axis = 0) # (6, 6)

    epsilon_star = np.linalg.inv(A) @ b

    T_op = pylgmath.se3.operations.vec2tran(epsilon_star) @ T_op

    print(T_op)