import os
import pickle
from datetime import datetime
from typing import Optional, List, Tuple
from copy import deepcopy

import numpy as np
import sim


class StereoLocalizationProblem:
    def __init__(
        self,
        T_wc: np.array,
        p_w: np.array,
        M: np.array,
        W: Optional[np.array] = None,
        y: Optional[np.array] = None,
        gamma_r: float = 0.0,
        r_0: Optional[np.array] = None,
        gamma_C: float = 0.0,
        C_0: Optional[np.array] = None,
        T_init: Optional[np.array] = None,
    ):
        self.T_wc = T_wc
        self.p_w = p_w
        self.y = y
        self.W = W
        self.M = M
        self.gamma_r = gamma_r
        self.r_0 = r_0
        self.gamma_C = gamma_C
        self.C_0 = C_0
        self.T_init = T_init

class StereoLocalizationSolution:

    def __init__(self, solved: bool, T_cw: Optional[np.array] = None, cost: Optional[float] = None, T_cw_history: Optional[List[np.array]] = None):
        self.solved = solved
        self.T_cw = T_cw
        self.cost = cost
        self.T_cw_history = T_cw_history

def make_sim_instances(num_instances: int, num_landmarks: int, p_wc_extent: np.array, cam: sim.Camera) -> List[Tuple[np.array, np.array]]:
    instances = []
    for _ in range(num_instances):
        world = sim.World(
            cam = cam,
            p_wc_extent = p_wc_extent,
            num_landmarks = num_landmarks,
        )
        world.clear_sim_instance()
        world.make_random_sim_instance()
        instances.append(StereoLocalizationProblem(world.T_wc, world.p_w, cam.M()))
    return instances


def run_experiment(metrics_fcn, var_list, num_problem_instances, num_landmarks, num_local_solve_tries, cam, p_wc_extent, W = None, r0 = None, gamma_r = 0, problem_instances = None):

    exp_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, f"outputs/{exp_time}")
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    instances = make_sim_instances(num_problem_instances, num_landmarks, p_wc_extent, cam)

    world = sim.World(
        cam = cam,
        p_wc_extent = p_wc_extent,
        num_landmarks = num_landmarks,
    )

    metrics = []

    for var in var_list:
        print(f"Noise Variance: {var}")
        world.cam.R = var * np.eye(4)
        for scene_ind in range(num_problem_instances):
            print(f"Scene ind: {scene_ind}")

            problem = instances[scene_ind]
            problem.y = world.cam.take_picture(problem.T_wc, problem.p_w)
            if W is None:
                problem.W = (1/var)*np.eye(4)
            else:
                problem.W = W
            problem.r_0 = r0
            problem.gamma_r = gamma_r

            for _ in range(num_local_solve_tries):
                T_op = sim.generate_random_T(p_wc_extent)
                problem.T_init = T_op
                metrics.append(metrics_fcn(problem))
                metrics[-1]["world"] = deepcopy(world)
                metrics[-1]["problem"] = deepcopy(problem)
                metrics[-1]['noise_var'] = var
                metrics[-1]['scene_ind'] = scene_ind

    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    return metrics, exp_dir
