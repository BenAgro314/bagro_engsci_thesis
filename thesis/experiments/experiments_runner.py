import os
import pickle
from typing import Callable

from thesis.datasets.dataset import (StereoLocalizationDataset,
                              StereoLocalizationExample)
from thesis.experiments.utils import make_and_get_experiment_dir


def run_experiment(dataset: StereoLocalizationDataset, metrics_fcn: Callable):
    metrics = []
    exp_dir = make_and_get_experiment_dir()
    with open(os.path.join(exp_dir, f"dataset_{dataset.name}_config.pkl"), 'wb') as f:
        pickle.dump(obj = dataset.config, file = f)

    for i, example in enumerate(dataset):
        print(f"Progress: [{i}/{len(dataset)}]")
        metrics.append(metrics_fcn(example))


    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    return metrics, exp_dir
