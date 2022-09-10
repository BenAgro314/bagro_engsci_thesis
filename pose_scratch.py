#%% imports 
import numpy as np
import pylgmath
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation

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
    x = C @ np.array([[1], [0], [0]])
    y = C @ np.array([[0], [1], [0]])
    z = C @ np.array([[0], [0], [1]])

    tag = Annotation3D(name, r, fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')
    ax.add_artist(tag)

    for c, v in zip(['r', 'g', 'b'], [x, y, z]):
        v = v.reshape(-1)
        end = r + v
        a = Arrow3D([r[0], end[0]], [r[1], end[1]],
                        [r[2], end[2]], mutation_scale=10, 
                        lw=2, arrowstyle="-|>", color=c)
        ax.scatter3D(end[0], end[1], end[2], s = 0)
        ax.add_artist(a)


#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

add_coordinate_frame(np.eye(4), ax, "$\mathfrak{F}_w$")

a = np.array([[1], [0], [0]])

C_cw = pylgmath.so3.operations.vec2rot(np.pi/4 * a/np.linalg.norm(a))
T_cw = np.eye(4)
T_cw[:3, :3] = C_cw
T_cw[:-1, -1] = [2, 3, 0]

add_coordinate_frame(T_cw, ax, "$\mathfrak{F}_c$")

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
#ax.xaxis.set_ticklabels([])
#ax.yaxis.set_ticklabels([])
#ax.zaxis.set_ticklabels([])

#scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']); ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
#N = 20

#data = np.random.rand(N, 3) + np.array([1, 1, 0])
#ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
#
#a = Arrow3D([2, 1], [2, 1], 
#                [0, 1], mutation_scale=10, 
#                lw=2, arrowstyle="-|>", color="r")
#ax.add_artist(a)

