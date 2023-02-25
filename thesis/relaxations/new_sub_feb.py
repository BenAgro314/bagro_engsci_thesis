#%%
# Imports
import sys
from ncpol2sdpa import *

sys.path.append("/home/agrobenj/bagro_engsci_thesis/thesis/")
sys.path.append("/Users/benagro/bagro_engsci_thesis/thesis/")
sys.path.append("/Users/benagro/bagro_engsci_thesis/")
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from thesis.simulation.sim import render_camera_points, World, Camera
import thesis.solvers.local_solver as local_solver
from thesis.solvers.global_sdp import global_sdp_solution
from thesis.solvers.local_solver import projection_error, StereoLocalizationProblem
from thesis.relaxations.sdp_relaxation_v2 import (
    build_general_SDP_problem,
    build_cost_matrix,
    build_rotation_constraint_matrices,
    build_measurement_constraint_matrices,
    build_parallel_constraint_matrices,
    build_homo_constraint,
    extract_solution_from_X,
)
import tikzplotlib
import scipy
import scipy.io


#%%

