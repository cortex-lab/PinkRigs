
#%%
import sys,re
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.data_manager import load_cluster_info


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = 'AV025',expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})


# %%
from predChoice import format_av_trials,glmFit

rec = recordings.iloc[4]
# preselect clusters based on quality metrics
# 
# 
#  
ev,spk,_,_,_ = simplify_recdat(rec,probe='probe')
clusInfo = load_cluster_info(rec,probe='probe')


my_ROI = 'SCm'
event_type = 'timeline_audPeriodOn'


from Processing.pyhist.helpers.regions import BrainRegions
from Analysis.neural.utils.spike_dat import bombcell_sort_units
br = BrainRegions()
bc_class = bombcell_sort_units(clusInfo)
clusInfo['is_good'] = bc_class=='good'
clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void' # this is just so that the berylacronymconversion does something good
clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')
goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym=='SCm')]._av_IDs.values

trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,t=0.15,onset_time=event_type)
# iterative fitting for each nrn 
nrn_IDs = [re.split('_',i)[1] for i in trials.columns if 'neuron' in i]

non_neural = trials.iloc[:,:3]
neural = trials.iloc[:,3:]
glm = glmFit(non_neural,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])
glm.fitCV(n_splits=2,test_size=0.5)

n_neurons = neural.shape[1]

thr = 0.005
best_nrn,ll_best = [],[]
ll_best = [glm.model.LogLik]
for i in range(n_neurons):
    print('finding #%.0f best neuron' % i)
    if i==0:
        base_matrix = non_neural
        bleed_matrix = neural
    else:
        base_matrix = pd.concat((non_neural,neural.loc[:,best_nrn]),axis=1)
        leftover_nrn = np.setdiff1d(neural.columns.values,np.array(best_nrn))
        bleed_matrix = neural.loc[:,leftover_nrn]
        
    ll = []
    for idx,(neuronName,trial_activity) in enumerate(bleed_matrix.iteritems()):
        fittable = pd.concat((base_matrix,trial_activity),axis=1)
        neuralglm = glmFit(fittable,model_type='AVSplit',fixed_parameters = [0,0,0,0,1,0],fixed_paramValues = list(glm.model.allParams))
        neuralglm.fitCV(n_splits=2,test_size=0.5)

        ll_current = neuralglm.model.LogLik
        if np.isnan(ll_current) or np.isinf(ll_current):
            ll_current= 1000
        ll.append(ll_current)
    
    curr_best_ll = np.min(np.array(ll))
    if curr_best_ll>(ll_best[i]-thr):
        print('the situation is not improving, you got to break...')
        break 
    ll_best.append(curr_best_ll)
    best_nrn.append(bleed_matrix.columns.values[np.argmin(np.array(ll))])

#%% 
# look at the metrics
import matplotlib.pyplot as plt
# refit and assess model contribution of each parameter
fig,ax = plt.subplots(1,1,figsize=(8,8))
final_matrix = pd.concat((non_neural,neural.loc[:,best_nrn]),axis=1)
final_glm = glmFit(final_matrix,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = list(glm.model.allParams))   
final_glm.fitCV(n_splits=2,test_size=0.5)
final_glm.visualise(yscale='sig',ax=ax)
fig.suptitle('{subject}_{expDate}_{expNum}_{probeID}'.format(**rec))
ax.set_title('LogLik=%.2f' % final_glm.model.LogLik)
#%%
# compare the neural vs non-neural models

#%%
fig,ax = plt.subplots(1,1,figsize=(8,8))

final_glm.plotPrediction(yscale='log',ax=ax)
ax.axline((0,0),slope=1,color='k',linestyle='--')
ax.set_xlabel('actual')
ax.set_ylabel('predicted')
ax.set_title('LogOdds')
# %%
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(ll_best)
ax.set_xlabel('# best neuron')
ax.set_ylabel('-Log2Likelihood')
ax.axhline(final_glm.model.LogLik,color='k',ls='--')
ax.text((len(ll_best)-5),final_glm.model.LogLik+0.004,'refit with all %.0f neurons' % (len(ll_best)-1))
ax.set_title('improvement on prediction from SC neuron prior to %s' % event_type)

fig,ax = plt.subplots(2,1,figsize=(25,5))
gParams = list(final_glm.model.required_parameters.copy())
ax[0].plot(gParams,final_glm.model.allParams[:6],'o-')
ax[1].plot(best_nrn,final_glm.model.allParams[6:])
ax[1].axhline(0,color='k',ls='--')
ax[1].set_xlabel('neuronID') 
#%%
fig,ax = plt.subplots(2,3,figsize=(15,8))
for i in range(2):

    if i==0:
        mytitle ='non-neural'
        m = non_neural
    elif i==1: 
        mytitle ='with neurons'
        m = final_matrix

    testglm = glmFit(m,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = list(glm.model.allParams))
    testglm.fitCV(n_splits=2,test_size=0.5)
    testglm.visualise(yscale='log',ax = ax[0,i])
    ax[0,i].set_title('%s, LogLik: %.2f' % (mytitle,testglm.model.LogLik))
    testglm.plotPrediction(yscale='sig',ax=ax[1,i])
    ax[1,2].hist(testglm.model.get_logOdds(testglm.conditions,testglm.model.allParams),alpha=0.5,range=(-8,8)) 
ax[1,2].set_xlabel('LogOdds')
ax[1,2].set_ylabel('# trials')



#%%
# plt.plot((final_matrix['choice']-.5)*2)
# plt.plot(final_matrix['neuron_135'])
# plt.xlabel('trial #')
# %%
