#%% 

import sys 
import numpy as np
from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params
import matplotlib.pyplot  as plt

recordings = load_rec_df(recompute_session_selection=False)
timing_params = load_params(paramset='choice')
#%%
from Admin.csv_queryExp import format_events,simplify_recdat
from predChoice import format_av_trials

rec = recordings.iloc[8]
ev,_,_,_,cam = simplify_recdat(rec,probe='probe')
trials = format_av_trials(ev,spikes=None,cam =cam,**timing_params)

non_movement_columns = [c for c in trials.columns if 'movement' not in c]
without_movement = trials[non_movement_columns]
with_movement = trials



fig,ax = plt.subplots(2,1,figsize=(8,16))

glm = glmFit(without_movement,model_type='AVSplit',
                fixed_parameters = [0,0,0,0,0,0])  
# get all sorts of likelihoods
glm.fitCV(n_splits=2,test_size=0.5)
glm.visualise(yscale='log',ax = ax[0])


# with movement 
glm = glmFit(with_movement,model_type='AVSplit',
                fixed_parameters = [0,0,0,0,0,0])        
glm.fitCV(n_splits=2,test_size=0.5)
glm.visualise(yscale='log',ax = ax[1])

# %%
fig,ax = plt.subplots(1,1,figsize=(25,2.5))
gParams = list(glm.model.required_parameters.copy())
gParams.append('movement')
ax.plot(gParams,glm.model.allParams,'o-')
ax.axhline(0,color='k',ls='--')
ax.set_xlabel('neuronID')
# %%
