#%% 

import sys 
import numpy as np
from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params

recordings = load_rec_df(recompute=True)
timing_params = load_params(paramset='choice')
#%%
from Admin.csv_queryExp import format_events,simplify_recdat
from predChoice import format_av_trials

rec = recordings.iloc[0]
ev,_,_,_,cam = simplify_recdat(rec,probe='probe')
trials = format_av_trials(ev,spikes=None,cam =cam,**timing_params)


with_movement = trials
# with movement 
glm = glmFit(with_movement,model_type='AVSplit',
                fixed_parameters = [0,0,0,0,0,0])        
glm.fitCV(n_splits=2,test_size=0.5)

# %%
