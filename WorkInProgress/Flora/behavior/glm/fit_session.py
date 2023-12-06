
#%%
import sys
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

#goodclusIDs = [158,228,122]
#%%
trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,t=0.1,onset_time='timeline_choiceMoveOn')


# %%
import matplotlib.pyplot as plt 
fig,ax = plt.subplots(1,1,figsize=(8,8))
glm = glmFit(trials,model_type='AVSplit')
glm.fitCV(n_splits=2,test_size=0.3)
glm.visualise(yscale='sigmoid',ax=ax)
fig.suptitle('{subject}_{expDate}_{expNum}_{probeID}'.format(**rec))
ax.set_title('LogLik=%.2f' % glm.model.LogLik)
# %%
print(glm.model.paramFit)

#%% 
import re

fig,ax = plt.subplots(2,1,figsize=(15,5))
gParams = list(glm.model.required_parameters.copy())
gParams.insert(0,'lassoLambda')
ax[0].plot(gParams,glm.model.paramFit[:7],'o-')
nrn_IDs = [re.split('_',i)[1] for i in trials.columns if 'neuron' in i]
ax[1].plot(nrn_IDs,glm.model.paramFit[7:])
ax[1].axhline(0,color='k',ls='--')
ax[1].set_xlabel('neuronID')
#%%
print(glm.model.LogLik)
# %%
