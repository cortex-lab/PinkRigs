# %%
import sys 
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.data_manager import load_cluster_info
from predChoice import glmFit,search_for_neural_predictors


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = ['AV025','AV030','AV034'],expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})

# %%
n_trials  = np.array([((rec.events._av_trials.response_direction>0) & (rec.events._av_trials.is_validTrial)).sum() for _,rec in recordings.iterrows()])

recordings = recordings.iloc[n_trials>100]
# %%


model_types = ['non-neural','neural']

logLiks = np.zeros((len(model_types),len(recordings)))
n_neurons = []

for r,(_,rec) in enumerate(recordings.iterrows()):

    trials = search_for_neural_predictors(rec,my_ROI='SCm',event_type = 'timeline_choiceMoveOn',ll_thr = 0.01)
    for i,model in enumerate(model_types):
        if model=='non-neural':
            trial_matrix = trials.iloc[:,:3]
        elif model=='neural': 
            trial_matrix = trials
            n_neurons.append(trials.shape[1]-3)

        glm = glmFit(trial_matrix,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])        
        glm.fitCV(n_splits=2,test_size=0.5)
        logLiks[i,r] = glm.model.LogLik




# %%
import matplotlib.pyplot as plt

plt.plot(logLiks,ls='-',color='k',alpha=.3,lw=8)
plt.plot(np.nanmean(logLiks,axis=1),color='k',lw=12)
plt.ylim([0.1,1])
#%%
plt.hist((logLiks[0]-logLiks[1])/logLiks[0],bins=20)


# %%
