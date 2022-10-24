from re import I
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

def plot_minimum_eigenvalues(metrics, path):
    # metrics[var][scene_ind][0, ..., num_local_solve_tries][{problem, solution, certificate}]

    for var in metrics:
        for scene_ind in metrics[var]:
            num_tries = len(metrics[var][scene_ind])
            metrics[var][scene_ind] = [v for v in metrics[var][scene_ind] if v["solution"].solved]
            if len(metrics[var][scene_ind]) == 0:
                continue
            metrics[var][scene_ind].sort(key = lambda x: x["solution"].cost)    
            npts = len(metrics[var][scene_ind])
            min_cost = metrics[var][scene_ind][0]["solution"].cost 
            colors = ["b" if np.isclose(v["solution"].cost, min_cost, rtol = 0, atol = 1e-3) else "r" for v in metrics[var][scene_ind]]
            plt.scatter([var] * npts, [min(v["certificate"].eig_values.real) for v  in metrics[var][scene_ind]], color = colors)
            print(f"Percentage Solved: {len(colors)/num_tries}")
            percent_global = colors.count('b')/len(colors)
            print(f"Percentage Of Solved that Are Global Solutions: {percent_global}")
            plt.annotate(f"{percent_global:.2f}", xy = (var, 0.1), fontsize = 8)

    plt.yscale("symlog")
    plt.xscale("log")
    plt.ylabel("Log of minimum eigenvalue from local solver")
    plt.xlabel("Pixel space gaussian measurement variance")
    plt.savefig(path)
    plt.show()
    plt.close("all")