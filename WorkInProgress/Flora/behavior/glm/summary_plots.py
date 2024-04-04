#%%
import pandas as pd
from pathlib import Path
from itertools import product

brain_areas = ['SCs','SCm','MRN','RSPd']
paramsets = ['prestim','poststim','choice']

[]

def load_results(brain_area,paramset):
    """
    wrapper to load the LogLikelihood results 
    """
    savepath= Path(r'D:\ChoiceEncoding\LogLikeilihoods_%s_%s.csv' % (brain_area,paramset))
    df = pd.read_csv(savepath)
    df['region'] = brain_area
    df['trial_period'] = paramset

    return df


logLiks = pd.concat([
    load_results(x[0],x[1]) for x in product(brain_areas,paramsets)
])



# %%
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

type = 'total'
which_lls = ['non-neural_%s' % type,'neural_%s' % type]

n_areas = len(brain_areas)
n_sets = len(paramsets)
fig,ax = plt.subplots(n_areas,n_sets,figsize=(4*n_sets,4*n_areas),sharey=True)

for i_area,area in enumerate(brain_areas):
    for i_set,set in enumerate(paramsets):
        df = logLiks[(logLiks.region==area) & (logLiks.trial_period==set)]


        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        mean_df = df.groupby('subject').mean().reset_index()

        _,p_value = stats.ttest_rel(mean_df[which_lls[0]], mean_df[which_lls[1]])



        ax[i_area,i_set].plot(
            mean_df[which_lls].values.T,
            ls='-',color='grey',alpha=.7,lw=1
            )
        ax[i_area,i_set].plot(
            mean_df[which_lls].values.T.mean(axis=1),
            ls='-',color='k',alpha=1,lw=2
            )
        
        #plt.ylim([-0.1,2])
        ax[i_area,i_set].set_title('p = %.4f' % p_value)

        if i_set==0:
            ax[i_area,i_set].set_ylabel(area)


which_figure = 'LogOdds_summary'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
# maybe calculate the differences and 
        

types = ['blank','aud','vis','coh','conf']   
fig,ax = plt.subplots(n_areas,n_sets,figsize=(3*n_sets,3*n_areas),sharey=True)


for i_area,area in enumerate(brain_areas):
    for i_set,set in enumerate(paramsets):
        df = logLiks[(logLiks.region==area) & (logLiks.trial_period==set)]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        mean_df = df.groupby('subject').mean().reset_index()

         
        ll_per_type = np.concatenate([(mean_df['neural_%s' % t]/mean_df['non-neural_%s' % t]).values[np.newaxis,:] for t in types])

        ax[i_area,i_set].plot(
            ll_per_type,
            ls='-',color='grey',alpha=.7,lw=1
            )
        ax[i_area,i_set].plot(
            np.median(ll_per_type,axis=1),
            ls='-',color='r',alpha=1,lw=2
            )
        
        ax[i_area,i_set].axhline(1,color='k',linestyle='--')
        
        ax[i_area,i_set].set_xticks(range(0,len(types)))
        ax[i_area,i_set].set_xticklabels(types)
        ax[i_area,i_set].set_ylim([0,2])

        if i_set==0:
            ax[i_area,i_set].set_ylabel(area)
# %%
