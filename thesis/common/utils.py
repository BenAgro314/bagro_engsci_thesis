import thesis
import os

def get_top_module_path():
    return os.path.dirname(thesis.__file__)

def get_data_dir_path():
    return os.path.join(get_top_module_path(), "datasets/data/")

def get_output_dir_path():
    path = os.path.join(get_top_module_path(), "experiments/outputs/")
    if not os.path.isdir(path):
        os.mkdir(path)
    return path