import numpy as np
from typing import Tuple, Union, Optional
import pylgmath
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from pylgmath.so3.operations import vec2rot
from . import plotting 

def make_stereo_sim_instance(num_points: int, T_wc: np.array, FOV: np.array):
    """
    Make a simulation instance for the stereo localization problem

    Args:
        num_points (int): Number of landmark points to add in the camera FOV
        T_wc (np.array): Rigid transformation from the camera frame to the world frame. Shape (4, 4)
        FOV (np.array): Shape (3, 2), defining
            [[x_min, x_max],
             [y_min, y_max],
             [z_min, z_max]]
            for the generated points in the camera frame
    Returns:
        1. matplotlib figure
        2. matplotlib axes
        3. Homogenous reference points in the world frame (N, 4, 1)
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plotting.add_coordinate_frame(np.eye(4), ax, "$\mathfrak{F}_w$")
    plotting.add_coordinate_frame(T_wc, ax, "$\mathfrak{F}_c$")
    # set aspect ratio

    ranges = FOV[:, 1] - FOV[:, 0]
    # generate points
    p_c = np.random.rand(num_points, 3) * ranges + FOV[:, 0] # (N, 3), camera frame points
    homo_p_c = np.concatenate((p_c, np.ones_like(p_c[:, 0:1])), axis = 1) # (N, 4, 1)
    p_w = T_wc @ homo_p_c[:, :, None] # (N, 4, 1), world frame points

    colors = np.random.rand(num_points, 3)
    colors = np.concatenate((colors, np.ones_like(colors[:, 0:1])), axis = 1)
    for i, p in enumerate(p_w):
        ax.scatter3D(p[0], p[1], p[2], color = colors[i])
    
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

    return fig, ax, p_w, colors

def make_stereo_camera_matrix(f_u: int, f_v: int, c_u: int, c_v:int, b: float) -> np.array:
    """
    Args:
        f_u (int): Focal length in horizontal pixels
        f_v (int): Focal length in vertical pixels
        c_u (int): Pinhole projection location in horizontal pixels
        c_v (int): Pinhole projection location in vertical pixels
        b (float): baseline (meters)

    Returns:
        np.array: Stereo camera parameter matrix (4,4)
    """
    return np.array(
        [
            [f_u, 0, c_u, f_u * b / 2],
            [0, f_v, c_v, 0],
            [f_u, 0, c_u, -f_u * b / 2],
            [0, f_v, c_v, 0],
        ]
    )
    

def generative_camera_model(M: np.array, T_cw: np.array, homo_p_w: np.array) -> np.array:
    """
    Args:
        M (np.array): Stereo camera parameters in a matrix
        T_cw (np.array): Rigid transform from world to camera frame
        homo_p_w (np.array): homogenous points in world frame

    Returns:
        y (np.array): (N, 4, 1), points in image space
            y[:, :2] are points in left image (row, col), indexed from the top left
            y[:, 2:] are points in right image (row, col), indexed from the top left
    """
    p_c = T_cw @ homo_p_w
    #assert np.all(p_c[:, 2] > 0)
    return M @ p_c / p_c[:, None, 2]

def generate_stereo_camera_noise(R: np.array, size: Optional[Union[Tuple, int]] = None):
    assert len(R.shape) == 2
    n = R.shape[0]
    assert R.shape == (n, n)
    return np.random.multivariate_normal(np.zeros(n), R, size = size)

def render_camera_points(y: np.array, colors: np.array):
    """
    Renders camera points stored in y to a matplotlib plot of the left and right images

    Args:
        y (np.array): (N, 4, 1), points in image space
            y[:, :2] are points in left image (row, col), indexed from the top left
            y[:, 2:] are points in right image (row, col), indexed from the top left
        colors (np.array): (N, 4), the colors associated with each point (for visual correspondence checking)
    """
    camera_points = y.squeeze(-1)
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

    return camfig, (lax, rax)