#%%
from ncpol2sdpa import *
import numpy as np


#%%

n_vars = 2 # Number of variables
level = 2  # Requested level of relaxation
x = generate_variables('x', n_vars)
obj = x[0]*x[1] + x[1]*x[0]
#inequalities = [-x[1]**2 + x[1] + 0.5>=0]
equalities = [x[0]**2 - x[0]]
sdp = SdpRelaxation(x)
sdp.get_relaxation(level, objective=obj, inequalities=inequalities,
                   equalities=equalities)
sdp.write_to_file("test.task")
#sdp.solve(solver = "mosek")
#%%

#import cvxpy as cp
#prob.solve(solver='MOSEK', mosek_params={'task_file_path': '/Users/benagro/bagro_engsci_thesis/thesis/experiments/test.task'})
import mosek
from mosek import Task
task = Task()
try:
    task.readdata("/Users/benagro/bagro_engsci_thesis/thesis/experiments/test.task")
    task.optimize()
except mosek.Error:
    print("Problem reading the file")

#%%