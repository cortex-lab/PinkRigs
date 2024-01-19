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
                        region_selection={'region_name':'VIS','min_fraction':.3})

# %%
n_trials  = np.array([((rec.events._av_trials.response_direction>0) & (rec.events._av_trials.is_validTrial)).sum() for _,rec in recordings.iterrows()])

recordings = recordings.iloc[n_trials>100]
# %%


model_types = ['non-neural','neural']

logLiks = np.zeros((len(model_types)+1,len(recordings)))
n_neurons = []
ll_nobias = []
for r,(_,rec) in enumerate(recordings.iterrows()):

    trials,gIDs = search_for_neural_predictors(rec,my_ROI='VISp',event_type = 'timeline_choiceMoveOn',ll_thr = 0.01)
    for i,model in enumerate(model_types):
        if model=='non-neural':
            trial_matrix = trials.iloc[:,:3]
        elif model=='neural': 
            trial_matrix = trials
            n_neurons.append(gIDs.size)

        glm = glmFit(trial_matrix,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])        
        glm.fitCV(n_splits=2,test_size=0.5)

        logLiks[i,r] = glm.model.LogLik

        if model=='neural':
            nobias_glm = glmFit(trial_matrix,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,1],fixed_paramValues=[1,1,1,1,1,0])   
            nobias_glm.fitCV(n_splits=2,test_size=0.5)
            logLiks[i+1,r] = (nobias_glm.model.LogLik)

# %%
import matplotlib.pyplot as plt

plt.plot(logLiks,ls='-',color='grey',alpha=1,lw=5)
#plt.plot(np.nanmean(logLiks,axis=1),color='k',lw=12)
plt.ylim([0.1,1])
plt.ylabel('-Log2Likelihood')
#%%
plt.plot(n_neurons,(logLiks[0]-logLiks[1]),'.',markersize=25,color='cyan',markeredgecolor='k')
plt.xlabel('#SC neurons available')
plt.ylabel('delta(LogLik)')
# %%
plt.hist(logLiks[0]-logLiks[1],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='k',histtype='step')
plt.hist(logLiks[0]-logLiks[2],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='g',histtype='step')
plt.hist(logLiks[1]-logLiks[2],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='r',histtype='step')
# %%
