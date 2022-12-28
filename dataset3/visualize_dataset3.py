#%% imports
import numpy as np
import scipy.io

#%% load dataset

dataset3 = scipy.io.loadmat('dataset3.mat')

#%% visualize ground truth

theta_vk_i = dataset3["theta_vk_i"]
r_i_vk_i = dataset3["r_i_vk_i"]

rho_i_pj_i = dataset3["rho_i_pj_i"]
# %%

