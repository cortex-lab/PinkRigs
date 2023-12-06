
#%%
import sys,re
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.data_manager import load_cluster_info


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = 'AV030',expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})


# %%
from predChoice import format_av_trials,glmFit

rec = recordings.iloc[2]
# preselect clusters based on quality metrics 
ev,spk,_,_,_ = simplify_recdat(rec,probe='probe')
clusInfo = load_cluster_info(rec,probe='probe')

# could build a helper for this
from Processing.pyhist.helpers.regions import BrainRegions
from Analysis.neural.utils.spike_dat import bombcell_sort_units
br = BrainRegions()
bc_class = bombcell_sort_units(clusInfo)
clusInfo['is_good'] = bc_class=='good'
clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void' # this is just so that the berylacronymconversion does something good
clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')

goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym=='SCm')]._av_IDs.values

goodclusIDs = [158,228,122]
#%%
trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,t=0.1,onset_time='timeline_choiceMoveOn')
# iterative fitting for each nrn 
nrn_IDs = [re.split('_',i)[1] for i in trials.columns if 'neuron' in i]

non_neural = trials.iloc[:,:3]
neural = trials.iloc[:,3:]
glm = glmFit(non_neural,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])
glm.fitCV(n_splits=2,test_size=0.5)

# fit all the neurons 1 by one

#%%
n_neurons = neural.shape[1]

best_nrn,ll_best = [],[]
ll_best = [glm.model.LogLik]
for i in range(n_neurons):
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
        neuralglm = glmFit(fittable,model_type='AVSplit',fixed_parameters = [1,1,1,1,1,1],fixed_paramValues = list(glm.model.allParams))
        neuralglm.fitCV(n_splits=2,test_size=0.5)
        ll.append(neuralglm.model.LogLik)
    
    curr_best_ll = np.min(np.array(ll))
    if curr_best_ll>ll_best[i]:
        print('the situation is not improving, you got to break...')
    ll_best.append(curr_best_ll)
    best_nrn.append(bleed_matrix.columns.values[np.argmin(np.array(ll))])



    

# # %%
# import matplotlib.pyplot as plt 
# fig,ax = plt.subplots(1,1,figsize=(8,8))

# glm = glmFit(trials.iloc[:,:4],model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = [1.5,1,1,1,1,0])
# #|
# glm.fitCV(n_splits=2,test_size=0.3)
# glm.visualise(yscale='sigmoid',ax=ax)
# fig.suptitle('{subject}_{expDate}_{expNum}_{probeID}'.format(**rec))
# ax.set_title('LogLik=%.2f' % glm.model.LogLik)
# # %%
# print(glm.model.paramFit)

# #%% 

# fig,ax = plt.subplots(2,1,figsize=(15,5))
# gParams = list(glm.model.required_parameters.copy())
# ax[0].plot(gParams,glm.model.allParams[:6],'o-')
# ax[1].plot(nrn_IDs,glm.model.paramFit[6:])
# ax[1].axhline(0,color='k',ls='--')
# ax[1].set_xlabel('neuronID')
# #%%
# print(glm.model.LogLik)
# # %%
