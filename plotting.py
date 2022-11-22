from re import I
from typing import List, Dict, Any
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
import matplotlib.pyplot as plt 

class Arrow3D(FancyArrowPatch):
    """Add an arrow with start and end coords to a 3d plot"""

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
    """Annotate the point xyz with text"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def add_coordinate_frame(T_wc: np.array, ax: plt.axes, label: str):
    """
    Adds a coordinate frame (x,y,z axes) to a 3d plot

    Args:
        T_wc (np.array): Rigid transformation from camera frame to world frame
        ax (plt.axes): The plt.axes to add the coordinate frame to
        label (str): The label to give the coordinate frame (supports Latex)
    """
    assert T_wc.shape == (4, 4)
    C = T_wc[:3, :3]
    r = T_wc[:-1, -1]
    #print(r)
    #print(C)
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(C @ C.T, np.eye(3))

    ax.scatter3D(r[0], r[1], r[2], s = 0)
    x = T_wc @ np.array([[1], [0], [0], [1]])
    y = T_wc @ np.array([[0], [1], [0], [1]])
    z = T_wc @ np.array([[0], [0], [1], [1]])

    tag = Annotation3D(label, r, fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')
    ax.add_artist(tag)

    for c, v in zip(['r', 'g', 'b'], [x, y, z]):
        v = v.reshape(-1)
        a = Arrow3D([r[0], v[0]], [r[1], v[1]],
                        [r[2], v[2]], mutation_scale=10, 
                        lw=2, arrowstyle="-|>", color=c)
        ax.scatter3D(v[0], v[1], v[2], s = 0)
        ax.add_artist(a)

def plot_minimum_eigenvalues(metrics: List[Dict[str, Any]], path: str):
    vars = [m["noise_var"] for m in metrics]
    vars_set = set(vars)
    scene_inds = [m["scene_ind"] for m in metrics]
    min_costs = {}
    for var in vars_set:
        min_costs[var] = {}
        for scene_ind in scene_inds:
            min_costs[var][scene_ind] = min([m["local_solution"].cost for m in metrics if (m["noise_var"] == var and m["scene_ind"] == scene_ind)])

    for m in metrics:
        if not m["local_solution"].solved:
            continue
        var = m["noise_var"]
        cost = m["local_solution"].cost
        scene_ind = m["scene_ind"]
        min_cost = min_costs[var][scene_ind]
        color = 'b' if np.isclose(min_cost, cost) else 'r'
        plt.scatter([var], min(m["certificate"].eig_values.real), color = color)

    plt.yscale("symlog")
    plt.xscale("log")
    plt.ylabel("Log of minimum eigenvalue from local solver")
    plt.xlabel("Pixel space gaussian measurement variance")
    plt.savefig(path)
    plt.show()
    plt.close("all")