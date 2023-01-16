import itertools
import sys
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import thesis.sim as sim
from thesis.experiments import StereoLocalizationProblem
from thesis.sim import Camera, World


@dataclass
class StereoLocalizationDatasetConfig():
    variances: List[float] # list of noise variances to generate examples with
    num_landmarks: List[int] # list of number of landmarks to put in examples
    num_examples_per_var_num_landmarks: int # number of examples to generate for each (variance, num_landmarks) pair
    num_T_inits_per_example: int # number of the same examples to add with different initial guesses,
    f_u: float 
    f_v: float
    c_u: float 
    c_v: float
    b: float
    fov_phi_range: Tuple[float, float]
    fov_depth_range: Tuple[float, float]
    p_wc_extent: np.array

@dataclass
class StereoLocalizationExample():
    problem: StereoLocalizationProblem
    camera: Camera
    world: World
    example_id: int # uniquely identifies scenarios generated with the same landmarks configs and pixel measurments
    
class StereoLocalizationDataset():

    def __init__(self):
        self.examples = []

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self.examples

    def add_example(self, example: StereoLocalizationExample):
        self.examples.append(example)

    @staticmethod
    def from_pickle(filepath: str):
        with open(filepath, "rb") as f:
            f.seek(0)
            return pickle.load(f)

    def to_pickle(self, filepath: str, force: bool = False):
        if not force:
            assert not os.path.exists(filepath), f"{filepath} already exists"
        with open(filepath, "wb") as f:
            return pickle.dump(self, f)

    def __getitem__(self, index: int):
        return self.examples[index]

    def __getstate__(self):
        return {"examples": self.examples}

    def __setstate__(self, state):
        self.examples = state["examples"]

    @staticmethod
    def from_config(config: StereoLocalizationDatasetConfig):
        dataset = StereoLocalizationDataset()
        example_id = 0
        for var, num_landmarks in itertools.product(config.variances, config.num_landmarks):
            print(f"On: variance: {var}, num_landmarks: {num_landmarks}")
            cam = Camera(
                f_u = config.f_u,
                f_v = config.f_v,
                c_u = config.c_u,
                c_v = config.c_v,
                b = config.b,
                R = var * np.eye(4),
                fov_phi_range = config.fov_phi_range,
                fov_depth_range = config.fov_depth_range,
            )
            world = sim.World(
                cam = cam,
                p_wc_extent = config.p_wc_extent,
                num_landmarks = num_landmarks
            )
            for _ in range(config.num_examples_per_var_num_landmarks):
                world.clear_sim_instance()
                world.make_random_sim_instance()
                y = cam.take_picture(world.T_wc, world.p_w)
                for _ in range(config.num_T_inits_per_example):
                    T_init = sim.generate_random_T(config.p_wc_extent)
                    example = StereoLocalizationExample(
                        problem = StereoLocalizationProblem(world.T_wc, world.p_w, cam.M(), y = y, T_init = T_init),
                        camera = deepcopy(cam),
                        world = deepcopy(world),
                        example_id = example_id,
                    )
                    dataset.add_example(example)
                example_id += 1
        return dataset

def main(out_path: str):
    np.random.seed(42)
    config = StereoLocalizationDatasetConfig(
        variances = [0.1, 0.3, 0.5, 0.7, 1],
        num_landmarks = [5, 10, 20],
        num_examples_per_var_num_landmarks = 10,
        num_T_inits_per_example = 100,
        f_u = 484.5,
        f_v = 484.5, 
        c_u = 322,
        c_v = 247,
        b = 0.24,
        fov_phi_range = (-np.pi / 12, np.pi / 12),
        fov_depth_range = (0.2, 3),
        p_wc_extent = np.array([[3], [3], [0]]),
    )
    dataset = StereoLocalizationDataset.from_config(config)
    dataset.to_pickle(out_path)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "python dataset.py <out_path>"
    out_path = sys.argv[1]
    main(out_path)
