import os
import random
from typing import List, Dict, Any
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from copy import deepcopy
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

def add_coordinate_frame(T_wc: np.array, ax: plt.axes, label: str, size: float = 1):
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
    x = T_wc @ np.array([[size], [0], [0], [1]])
    y = T_wc @ np.array([[0], [size], [0], [1]])
    z = T_wc @ np.array([[0], [0], [size], [1]])

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

def plot_local_and_iterative_compare(metrics: List[Dict[str, Any]], path: str):
    fig, axs = plt.subplots(2, 1)
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    local_solution_costs = [m["local_solution"].cost for m in metrics]
    iterative_sdp_costs = [m["iterative_sdp_solution"].cost for m in metrics]

    min_cost = min(min(local_solution_costs), min(iterative_sdp_costs))
    max_cost = max(max(local_solution_costs), max(iterative_sdp_costs))
    bins = np.logspace(np.log10(min_cost),np.log10(max_cost), 50)
    axs[0].hist(local_solution_costs, bins=bins)
    axs[0].set_xlabel("Local Solver Solution Cost")
    axs[1].hist(iterative_sdp_costs, bins=bins)
    axs[1].set_xlabel("Iterative SDP Solution Cost")
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(path)
    plt.show()
    plt.close("all")

def plot_min_cost_vs_noise(metrics: List[Dict[str, Any]], path: str):
    vars = [m["noise_var"] for m in metrics]
    vars = list(set(vars))
    vars.sort()
    scene_inds = [m["scene_ind"] for m in metrics]
    min_costs = {}

    for var in vars:
        min_costs[var] = {}
        for scene_ind in scene_inds:
            min_costs[var][scene_ind] = min([m["local_solution"].cost for m in metrics if (m["noise_var"] == var and m["scene_ind"] == scene_ind)])

    data = {
        "": [sum(list(min_costs[var].values()))/len(min_costs[var]) for var in vars],
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, tick_labels = vars)
    ax.set_ylabel("Minimum Cost")
    ax.set_xlabel("Pixel-space Noise Variance")

    #fig.subplots_adjust(hspace=0.5)
    plt.savefig(path)
    plt.show()
    plt.close("all")


def plot_percent_succ_vs_noise(metrics: List[Dict[str, Any]], path: str):

    vars = [m["noise_var"] for m in metrics]
    vars = list(set(vars))
    vars.sort()
    scene_inds = [m["scene_ind"] for m in metrics]
    min_costs = {}
    local_counts = {v: 0 for v in vars}
    iter_sdp_counts = {v: 0 for v in vars}
    totals = {v: 0 for v in vars}
    for var in vars:
        min_costs[var] = {}
        for scene_ind in scene_inds:
            min_costs[var][scene_ind] = min([m["local_solution"].cost for m in metrics if (m["noise_var"] == var and m["scene_ind"] == scene_ind)])


    for m in metrics:
        var = m["noise_var"]
        scene_ind = m["scene_ind"]
        local_cost = m["local_solution"].cost
        iterative_sdp_cost = m["iterative_sdp_solution"].cost
        local_counts[var] += 1 if np.isclose(local_cost, min_costs[var][scene_ind]) else 0
        iter_sdp_counts[var] += 1 if np.isclose(iterative_sdp_cost, min_costs[var][scene_ind]) else 0
        totals[var] += 1

    data = {
        "local solver": [local_counts[var]/totals[var] for var in vars],
        "iterative sdp": [iter_sdp_counts[var]/totals[var] for var in vars],
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, tick_labels = vars)
    ax.set_ylabel("Percentage of Globally Optimal Solutions")
    ax.set_xlabel("Pixel-space Noise Variance")
    ax.set_ylim([0, 1.1])

    #fig.subplots_adjust(hspace=0.5)
    plt.savefig(path)
    plt.show()
    plt.close("all")

def plot_solution_history(path: str, problem, solution, world):
    assert solution.T_cw_history is not None
    world = deepcopy(world)
    world.T_wc = problem.T_wc
    world.p_w = problem.p_w
    fig, ax, colors = world.render(include_world_frame = False)
    T_cw_history = solution.T_cw_history
    for i, T_cw in enumerate(T_cw_history):
        add_coordinate_frame(np.linalg.inv(T_cw), ax, "$\mathcal{F}" + f"_{i}$")
    fig.savefig(path)
    #plt.show()
    plt.close("all")

def plot_select_solutions_history(metrics: List[Dict[str, Any]], exp_dir: str, num_per_noise: int = 1):
    metrics_by_noise = {}
    for m in metrics:
        var = m["noise_var"]
        if var not in metrics_by_noise:
            metrics_by_noise[var] = [m]
            os.mkdir(os.path.join(exp_dir, str(var)))
        else:
            metrics_by_noise[var].append(m)

    for var in metrics_by_noise:
        ms = random.choices(metrics_by_noise[var], k = num_per_noise)
        for i, m in enumerate(ms):
            local_solution = m["local_solution"]
            iterative_sdp_solution = m["iterative_sdp_solution"]
            world = m["world"]
            problem = m["problem"]
            plot_solution_history(os.path.join(exp_dir, str(var), f"local_solution_{var}_{i}.png"), problem, local_solution, world)
            plot_solution_history(os.path.join(exp_dir, str(var), f"iterative_sdp_solution_{var}_{i}.png"), problem, iterative_sdp_solution, world)

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, tick_labels = None):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

            # Create annotation
            ax.annotate(
                y,                      # Use `label` as label
                (x+x_offset, y),         # Place label at end of the bar
                xytext=(0, 5),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va='bottom')                      # Vertically align label differently for
                                            # positive and negative values.

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])


    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

    if tick_labels is not None:
        ax.set_xticks(list(range(len(tick_labels))))
        ax.set_xticklabels([str(t) for t in tick_labels])

