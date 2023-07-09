#%% 
# code to implement CCCP (modified Mann-Whitney U-test) by Steinmetz et al 2019


#def choice_probability(spikeCount,choices,trialCondition):

import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.cccp import cccp,get_default_set

pars = get_default_set()

# loading
mname = 'AV030'
expDate = '2022-12-09'
probe = 'probe1'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
    'probe': probe
}

c = cccp()
c.load_and_format_data(**session)
# %%

u,p,_,t = zip(*[c.get_U(**cp) for _,cp in pars.iterrows()])
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
