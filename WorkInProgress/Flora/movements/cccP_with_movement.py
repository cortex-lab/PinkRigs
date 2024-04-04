#%%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from loaders import load_for_movement_correlation
recordings = load_for_movement_correlation(dataset='active',recompute_data_selection=True)


#%%
from Analysis.neural.src.cccp import cccp,get_default_set

pars = get_default_set(which='single_bin')

def run_all_ccCPs(rec,pars):
    c = cccp()
    c.load_and_format_data(rec=rec)
    c.aud_azimuths=[0]
    _,p,_,_ = zip(*[c.get_U(which_dat='video',**cp) for _,cp in pars.iterrows()])

    return p[0][0],p[1][0],p[2][0]


pA,pC,pV= zip(*[run_all_ccCPs(rec,pars) for _,rec in recordings.iterrows()])


# %% 
pA,pC,pV = np.array(pA),np.array(pC),np.array(pV)

# %%
(pV<0.05).sum()/pA.size
# %%
