from typing import Optional
import sys
import os
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Any

import numpy as np
from common.utils import get_data_dir_path

from thesis.experiments.utils import StereoLocalizationProblem
from thesis.simulation.sim import Camera, World, generate_random_T


@dataclass
class StereoLocalizationDatasetConfig():
    name: str
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
    world: Optional[World]
    example_id: Any # uniquely identifies scenarios generated with the same landmarks configs and pixel measurments
    
class StereoLocalizationDataset():

    def __init__(self, name: str):
        self.examples = []
        self.name = name
        self.config = None

    @staticmethod
    def from_config(config: StereoLocalizationDatasetConfig):
        examples = []

        cam = Camera(
            f_u = config.f_u,
            f_v = config.f_v,
            c_u = config.c_u,
            c_v = config.c_v,
            b = config.b,
            R = 1 * np.eye(4),
            fov_phi_range = config.fov_phi_range,
            fov_depth_range = config.fov_depth_range,
        )

        worlds = []

        # create world first so we can re-use the same landmark configs across noise levels
        for num_landmarks in config.num_landmarks:
            for _ in range(config.num_examples_per_var_num_landmarks):
                world = World(
                    cam = cam,
                    p_wc_extent = config.p_wc_extent,
                    num_landmarks = num_landmarks
                )
                world.clear_sim_instance()
                world.make_random_sim_instance()
                worlds.append(deepcopy(world))

        for world in worlds:
            for var in config.variances:
                world.cam.R = var * np.eye(4)
                y = world.cam.take_picture(world.T_wc, world.p_w)
                W = (1/var)*np.eye(4)
                for _ in range(config.num_T_inits_per_example):
                    T_init = generate_random_T(config.p_wc_extent)
                    example = StereoLocalizationExample(
                        problem = StereoLocalizationProblem(world.T_wc, world.p_w, cam.M(), y = y, T_init = T_init, W = W),
                        camera = deepcopy(world.cam),
                        world = deepcopy(world),
                        example_id = hash(str(world.T_wc) + str(world.p_w) + str(y)),
                    )
                    examples.append(example)

        dataset = StereoLocalizationDataset(config.name)
        dataset.examples = examples
        dataset.shuffle()
        dataset.config = config
        return dataset

    def __len__(self):
        return len(self.examples)

    def shuffle(self):
        random.shuffle(self.examples)

    def __iter__(self):
        return self.examples.__iter__()

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
        return {"examples": self.examples, "config": self.config, "name": self.name}

    def __setstate__(self, state):
        self.examples = state["examples"]
        self.config = state["config"]
        self.name = state["name"]

def main(dataset_name: str):
    config = StereoLocalizationDatasetConfig(
        name = dataset_name,
        variances = [0.1, 0.5, 0.7, 1.0],
        num_landmarks = [5],
        num_examples_per_var_num_landmarks = 2,
        num_T_inits_per_example = 25,
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
    dataset_path = os.path.join(get_data_dir_path(), dataset_name + ".pkl")
    dataset.to_pickle(dataset_path, force = True)
    print(f"Written dataset of lengh {len(dataset)}")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "python dataset.py <dataset_name>"
    dataset_name = sys.argv[1]
    main(dataset_name)
