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
import ipympl


#%% functions

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer = None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        #FancyArrowPatch.draw(self, renderer)

        return np.min(zs)

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

        #return np.min(zs)

def add_coordinate_frame(T_cw: np.array, ax: plt.axes, name: str):
    assert T_cw.shape == (4, 4)
    C = T_cw[:3, :3]
    r = T_cw[:-1, -1]
    #print(r)
    #print(C)
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(C @ C.T, np.eye(3))

    ax.scatter3D(r[0], r[1], r[2], s = 0)
    x = T_cw @ np.array([[1], [0], [0], [1]])
    y = T_cw @ np.array([[0], [1], [0], [1]])
    z = T_cw @ np.array([[0], [0], [1], [1]])

    tag = Annotation3D(name, r, fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')
    ax.add_artist(tag)

    for c, v in zip(['r', 'g', 'b'], [x, y, z]):
        v = v.reshape(-1)
        a = Arrow3D([r[0], v[0]], [r[1], v[1]],
                        [r[2], v[2]], mutation_scale=10, 
                        lw=2, arrowstyle="-|>", color=c)
        ax.scatter3D(v[0], v[1], v[2], s = 0)
        ax.add_artist(a)


#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

add_coordinate_frame(np.eye(4), ax, "$\mathfrak{F}_w$")

a1 = np.array([[1], [0], [0]])
a2 = np.array([[0], [0], [1]])
a3 = np.array([[1], [0], [0]])

C_cw = vec2rot(0 * a3/np.linalg.norm(a3)) @ vec2rot(3*np.pi/4 * a2/np.linalg.norm(a2)) @ vec2rot(-np.pi/2 * a1/np.linalg.norm(a1)) 
T_cw = np.eye(4)
T_cw[:3, :3] = C_cw
T_cw[:-1, -1] = [3, 3, 0]

add_coordinate_frame(T_cw, ax, "$\mathfrak{F}_c$")

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


# add points

N = 4

points = np.random.rand(N, 3) * np.array([1, 1, 2]) + np.array([1, 1, -1])
colors = {}
cmap = plt.cm.get_cmap("hsv", N)
for i, p in enumerate(points):
    colors[i] = cmap(i) 
    ax.scatter3D(p[0], p[1], p[2], color = colors[i])

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
    #assert np.all(p_c[:, 2] > 0)
    print(p_c)
    return M @ p_c / p_c[:, None, 2]

homo_points = np.concatenate((points, np.ones_like(points[:, 0:1])), axis = 1)[:, :, None]

camera_points = camera_model(M, np.linalg.inv(T_cw), homo_points).squeeze(-1)
left = camera_points[:, :2]
right = camera_points[:, 2:]

camfig, (lax, rax) = plt.subplots(1, 2)
lax.invert_yaxis()
#lax.invert_xaxis()
rax.invert_yaxis()
#rax.invert_xaxis()
for i, (pl, pr) in enumerate(zip(left, right)):
    lax.scatter(pl[0], pl[1], color = colors[i])
    rax.scatter(pr[0], pr[1], color = colors[i])
#lax.set_xlim(left = 0)
#lax.set_ylim(top = 0)
#rax.set_xlim(left = 0)
#rax.set_ylim(top = 0)
rax.set_aspect('equal')
lax.set_aspect('equal')

plt.show()
