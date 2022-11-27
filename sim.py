from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import json

import matplotlib.pyplot as plt
import numpy as np
import pylgmath
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D, proj3d
from pylgmath.so3.operations import vec2rot

import plotting


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

@dataclass
class Camera:

    f_u: int
    f_v: int
    c_u: int
    c_v: int
    b: float
    R: np.array # noise covariance matrix
    fov: np.array # field of view of camera

    def M(self):
        return make_stereo_camera_matrix(
            self.f_u,
            self.f_v,
            self.c_u, 
            self.c_v,
            self.b
        )

    def take_picture(self, T_wc: np.array, p_w: np.array) -> np.array:
        """Moves camera to world pose T_wc and takes a picture,
        with landmarks at p_w. Returns pixel positions of landmarks

        Args:
            T_wc (np.array): World pose of camera (4, 4)
            p_w (np.array): Position of landmarks in the world (N, 4, 1)

        Returns:
            np.array: np.array of shape (N, 4, 1)
        """
        T_cw = np.linalg.inv(T_wc)
        y = generative_camera_model(self.M(), T_cw, p_w)
        dy = generate_stereo_camera_noise(self.R, size = y.shape[0])[:, :, None]
        return y + dy

@dataclass
class World:

    cam: Camera
    p_wc_extent: np.array # (3, 1) --- x, y, z
    num_landmarks: int
    T_wc: Optional[np.array] = None
    p_w: Optional[np.array] = None

    def place_landmarks_in_camera_fov(self):
        assert self.T_wc is not None, "Need camera frame in world to place landmarks in camera FOV!"
        ranges = self.cam.fov[:, 1] - self.cam.fov[:, 0]
        # generate points
        p_c = np.random.rand(self.num_landmarks, 3) * ranges + self.cam.fov[:, 0] # (N, 3), camera frame points
        homo_p_c = np.concatenate((p_c, np.ones_like(p_c[:, 0:1])), axis = 1) # (N, 4, 1)
        self.p_w = self.T_wc @ homo_p_c[:, :, None] # (N, 4, 1), world frame points

    def clear_sim_instance(self):
        self.T_wc = None
        self.p_w = None

    def make_random_sim_instance(self):
        if self.T_wc is None:
            self.T_wc = generate_random_T(self.p_wc_extent)

        if self.p_w is None:
            self.place_landmarks_in_camera_fov()

    def render(self):
        assert self.T_wc is not None, "Need to generate sim instance first! see `make_random_sim_instance`"
        assert self.p_w is not None, "Need to generate sim instance first! see `make_random_sim_instance`"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plotting.add_coordinate_frame(np.eye(4), ax, "$\mathfrak{F}_w$")
        plotting.add_coordinate_frame(self.T_wc, ax, "$\mathfrak{F}_c$")
        # set aspect ratio

        colors = np.random.rand(self.num_landmarks, 3)
        colors = np.concatenate((colors, np.ones_like(colors[:, 0:1])), axis = 1)
        for i, p in enumerate(self.p_w):
            ax.scatter3D(p[0], p[1], p[2], color = colors[i])
        
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

        return fig, ax, colors

        
def generate_random_rot():
    a = np.random.rand(3, 1)
    theta = np.random.rand() * 2*np.pi
    return  vec2rot(theta * a/np.linalg.norm(a))

def generate_random_T(p_extent: np.array):
    C_wc = generate_random_rot()

    T = np.eye(4)
    T[:3, :3] = C_wc

    T[:-1, -1:] = p_extent * np.random.rand(3, 1)
    return T