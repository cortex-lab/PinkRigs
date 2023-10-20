# %%
import pyddm 
import time 
from fitting import get_parameters
from model_components import get_freeP_sets


freePs = get_freeP_sets('fixed')
fit_params = get_parameters(freePs=freePs)             
m = pyddm.Model(**fit_params)

t0= time.time()
[m.solve_analytical(conditions={'visDiff':0.0,'audDiff':0.0,'is_laserTrial':0.0}) for i in range(1000)]
print(time.time()-t0)


# t0= time.time()
# [m.solve_numerical(conditions={'visDiff':0.0,'audDiff':0.0,'is_laserTrial':0.0}) for i in range(1000)]
# print(time.time()-t0)
# %%
