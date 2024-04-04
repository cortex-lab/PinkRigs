#%%
import sys,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from predChoice import format_av_trials,glmFit


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = 'AV008',expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=False,merge_probes=True,
                        region_selection=None)




# %%

#%%
rec = recordings.iloc[7]
# preselect clusters based on quality metrics 
ev,spk,_,_,_ = simplify_recdat(rec,probe='probe')
trials = format_av_trials(ev,spikes=None)
glm = glmFit(trials,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])
glm.fit()
#glm.fitCV(n_splits=2,test_size=0.1)
#%%
glm.visualise(yscale='sig')

# %%
fig,ax = plt.subplots(1,1,figsize = (7,5))
window=20
tr = np.convolve(ev.response_direction==0,np.ones(window))
ax.plot(tr)
ax.axvline(np.argmax(tr)-window)


# to_keep_nogo = np.ones(ev.is_blankTrial.size).astype('bool')
# to_keep_nogo[np.argmax(tr):] = False 


# %%
# filter the ev  structure
window_size = 200
half_window = int(window_size/2)

params = []
for i in range((np.argmax(tr)-window)-window_size):
    to_keep = np.zeros(ev.is_blankTrial.size).astype('bool')
    x = half_window+i
    to_keep[(x-half_window):(x+half_window)] = True

    currev = Bunch({k:ev[k][to_keep] for k in ev.keys()})
    #print((currev.is_validTrial & (currev.response_direction>0)).sum())
    trials = format_av_trials(currev,spikes=None)
    trialglm = glmFit(trials,model_type='AVSplit',fixed_parameters = [0,0,0,0,1,0],fixed_paramValues=list(glm.model.allParams))
    trialglm.fit()
    #trialglm.visualise(yscale='log')
    params.append(trialglm.model.allParams[np.newaxis,:])



params = np.concatenate(params)
# %% 

fig,ax = plt.subplots(6,1,figsize=(3,20))

for i,p in enumerate(params.T):
    ax[i].plot(p)
    ax[i].set_title(glm.model.required_parameters[i])
    ax[i].axhline(glm.model.allParams[i],color='k',ls='--')
# %%
fig,ax = plt.subplots(1,1,figsize=(15,3))
ax.plot(ev.response_direction) 


df = pd.DataFrame(ev)

# 
to_keep = ev.is_validTrial & (ev.response_direction>0)

fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.plot(
    df['response_direction'][to_keep].rolling(25).mean()
)


# %%

my_i = 170
x = half_window+my_i
to_keep = np.zeros(ev.is_blankTrial.size).astype('bool')

to_keep[(x-half_window):(x+half_window)] = True

currev = Bunch({k:ev[k][to_keep] for k in ev.keys()})
#print((currev.is_validTrial & (currev.response_direction>0)).sum())
trials = format_av_trials(currev,spikes=None)
print(trials)
trialglm = glmFit(trials,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues=[1,1,1,1,1,1])
trialglm.fit()
trialglm.visualise(yscale='sig')


# %%
