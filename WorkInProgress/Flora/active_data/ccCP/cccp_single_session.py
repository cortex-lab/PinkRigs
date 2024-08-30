#%% 
# code to implement CCCP (modified Mann-Whitney U-test) by Steinmetz et al 2019


#def choice_probability(spikeCount,choices,trialCondition):

import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.cccp import cccp,get_default_set

pars = get_default_set(which='single_bin',t_length=0.2,t_bin=0.005)
# loading
mname = 'AV009'
expDate = '2022-03-03'
probe = 'probe0'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
}
from Admin.csv_queryExp import load_data

ephys_dict = {'spikes':'all','clusters':'all'}

current_params = {
'data_name_dict':{'probe0':ephys_dict,'probe1':ephys_dict,
                    'events': {'_av_trials': 'table'}},
                    'checkSpikes':'1',
                    'unwrap_probes':True,
                    'filter_unique_shank_positions':True,
                    'merge_probes':False,                              
                    'cam_hierarchy': None

}  
recordings = load_data(**session,**current_params)
rec = recordings.iloc[0]

c = cccp()
c.load_and_format_data(rec=rec)

c.aud_azimuths=[0]


u,p,_,t, = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])
#p = np.concatenate(p,axis=1)


# %%
# what we can represent
sI  =[]
for idx in range(len(p)):
    fig,ax = plt.subplots(1,1)
    ax.plot(t[idx],(p[idx]<0.01).mean(axis=0))

    n_sel = (p[idx]<0.01).sum(axis=1)
    c.clusters._av_IDs[np.argsort(n_sel)[::-1]]

    selIDs  = c.clusters._av_IDs[np.argsort(n_sel)[::-1]][(np.sort(n_sel)[::-1])>0]

    sI.append(selIDs)

# %%
