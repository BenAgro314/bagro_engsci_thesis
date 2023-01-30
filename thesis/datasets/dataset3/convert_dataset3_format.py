import scipy.io
import os
from pylgmath.so3.operations import hat
from copy import deepcopy
from common.utils import get_data_dir_path, get_top_module_path
from datasets.dataset import StereoLocalizationDataset, StereoLocalizationExample
from thesis.simulation.sim import Camera, generate_random_T
import numpy as np
from thesis.experiments.utils import StereoLocalizationProblem

def read_dataset(in_path: str, dataset_name: str, num_T_inits_per_example: int = 20):
    dataset3 = scipy.io.loadmat(in_path)

    # F_i: inertial frame
    # F_v: vehicle frame
    # F_c: camera frame

    theta_vk_i = dataset3["theta_vk_i"] # a 3xK matrix
    r_i_vk_i = dataset3["r_i_vk_i"] # a 3xK matrix where the kth column is the groundtruth position of the camera at timestep k

    y_k_j = dataset3["y_k_j"] # 4 x K x 20 array of observations. All components of y_k_j(:, k, j) will be -1 if the observation is invalid
    y_var = dataset3["y_var"] # 4 x 1 matrix of computed variances based on ground truth stereo measurements
    rho_i_pj_i = dataset3["rho_i_pj_i"] # a 3x20 matrix where the jth column is the poisition of feature j

    # camera to vehicle
    C_c_v = dataset3["C_c_v"] # 3 x 3 matrix giving rotation from vehicle frame to camera frame
    rho_v_c_v = dataset3["rho_v_c_v"]

    # intrinsics
    fu = dataset3["fu"]
    fv = dataset3["fv"]
    cu = dataset3["cu"]
    cv = dataset3["cv"]
    b = dataset3["b"]
    print(fu, fv, cu, cv, b)

    cam = Camera(fu, fv, cu, cv, b, R = np.diag(y_var.reshape(-1)), fov_phi_range=(0, 0), fov_depth_range=(0, 0)) # is this noise correct?
    M = np.array(cam.M(), dtype=np.float64)

    T_c_v = np.eye(4)
    T_c_v[:3, :3] = C_c_v
    T_c_v[:3, -1:] = -C_c_v @ rho_v_c_v

    problems = []

    p_max = -np.inf * np.ones((3, 1))
    W = np.array(np.linalg.inv(np.diag(y_var.reshape(-1))), dtype=np.float64)
    #W = np.ones((4, 4), dtype=np.float64)


    for k, psi in enumerate(theta_vk_i.T):

        p_w = rho_i_pj_i.T[:, :, None]
        p_w = np.concatenate((p_w, np.ones_like(p_w[:, 0:1, :])), axis = 1)
        ys_with_invalid = y_k_j[:, k, :].T[:, :, None]
        mask = ~((ys_with_invalid.squeeze(-1) == -1).all(axis = 1))
        p_w = np.array(p_w[mask] , dtype=np.float64)

        if p_w.shape[0] < 3: # reject problems with less than 3 landmark points in the image
            continue

        y = np.array(ys_with_invalid[mask], dtype=np.float64)

        psi = psi.reshape(3, 1)
        psi_mag = np.linalg.norm(psi)
        C_vk_i = np.cos(psi_mag) * np.eye(3) + ( 1 - np.cos(psi_mag) ) * (psi / psi_mag) @ (psi / psi_mag).T - np.sin(psi_mag) * hat(psi / psi_mag)
        T_vk_i = np.eye(4)
        T_vk_i[:3, :3] = C_vk_i
        T_vk_i[:3, -1] = - C_vk_i @ r_i_vk_i[:, k]
        T_ck_i = T_c_v @ T_vk_i

        T_w_ck = np.array(np.linalg.inv(T_ck_i), dtype = np.float64)
        problems.append(
            StereoLocalizationProblem(
                T_wc = T_w_ck,
                p_w = p_w,
                M = M,
                y = y,
                W = W,
            )
        )
        p_max = np.maximum(p_max, T_w_ck[:3, -1:])

    p_wc_extent = p_max

    dataset = StereoLocalizationDataset(dataset_name)

    for problem in problems:
        num_landmarks = problem.y.shape[0]
        if num_landmarks < 4:
            continue
        for _ in range(num_T_inits_per_example):
            T_init = generate_random_T(p_wc_extent)
            p = deepcopy(problem)
            p.T_init = T_init
            example = StereoLocalizationExample(
                problem = p,
                camera = cam,
                world = None,
                example_id = hash(str(problem.T_wc) + str(problem.p_w) + str(problem.y)),
            )
            dataset.add_example(example)


    dataset_path = os.path.join(get_data_dir_path(), dataset_name)
    print(len(dataset))
    dataset.to_pickle(dataset_path)

if __name__ == "__main__":
    in_path = os.path.join(get_top_module_path(), "datasets/dataset3/dataset3.mat")
    read_dataset(in_path, "starry_nights.pkl")