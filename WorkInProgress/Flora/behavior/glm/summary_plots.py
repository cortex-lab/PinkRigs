#%%
import sys

import pandas as pd
from pathlib import Path
from itertools import product

brain_areas = ['SCs','SCm','MRN','RSPd']
paramsets = ['prestim','poststim','choice']
import seaborn as sns 
from statsmodels.regression.mixed_linear_model import MixedLM


def load_results(brain_area,paramset):
    """
    wrapper to load the LogLikelihood results 
    """
    set_name = '%s_%s' % (brain_area,paramset)
    savepath= Path(r'D:\LogRegression\%s' % set_name)

    df_path = list(savepath.glob('result_*.csv'))
    df = pd.read_csv(df_path[-1])
    df['region'] = brain_area
    df['trial_period'] = paramset

    return df


logLiks = pd.concat([
    load_results(x[0],x[1]) for x in product(brain_areas,paramsets)
])

#%%


# %%
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

type = 'total'
which_lls = ['non-neural_%s' % type,'neural_%s' % type]

n_areas = len(brain_areas)
n_sets = len(paramsets)
fig,ax = plt.subplots(n_areas,n_sets,figsize=(2*n_sets,2*n_areas),sharey=True,sharex=True)



markers = {
  "AV005": "o", 
  "AV008": "s",
  "AV014": "D",
  "AV020": "^", 
  "AV025": "v",
  "AV030": "P", 
  "AV034": "H", 
  "FT030": "<",
  "FT032": "X",
  "FT035": ">"   
}

for i_area,area in enumerate(brain_areas):
    for i_set,set in enumerate(paramsets):
        df = logLiks[(logLiks.region==area) & (logLiks.trial_period==set)]

        ax[i_area,i_set].axline((0,0),slope=1,color='k',linestyle='--')

        sns.scatterplot(
            data=df,
            x='negLL_stim',y='negLL_all',
            style='subject',s=100,
            ax=ax[i_area,i_set],markers=markers,palette='magma_r',
            legend=True,c='cyan',edgecolor='k'
            )


        ax[i_area,i_set].set_xlim([0.45,1])
        ax[i_area,i_set].set_ylim([0.45,1])

        df_long = pd.melt(df, id_vars=['subject'], value_vars=['negLL_all', 'negLL_stim'],
                 var_name='condition', value_name='negLL')
        df_long['condition'] = df_long['condition'].astype('category')

        # Display the reshaped DataFrame
        model = MixedLM.from_formula("negLL ~ condition", df_long, groups=df_long["subject"])
        result = model.fit()
        pval = result.pvalues['condition[T.negLL_stim]']

        ax[i_area,i_set].set_title('%.3f' % pval)

fig.subplots_adjust(hspace=0.5, wspace=0.5)

handles, labels = ax[1, 2].get_legend_handles_labels()
# Remove individual legends
for a in ax.flatten():
    a.get_legend().remove()
fig.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.05, 0.5), title='Time')
plt.show()

which_figure = 'LogOdds_summary'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


# %%




# # %%

# # maybe calculate the differences and 
        

# types = ['blank','aud','vis','coh','conf']   
# fig,ax = plt.subplots(n_areas,n_sets,figsize=(3*n_sets,3*n_areas),sharey=True)


# for i_area,area in enumerate(brain_areas):
#     for i_set,set in enumerate(paramsets):
#         df = logLiks[(logLiks.region==area) & (logLiks.trial_period==set)]

#         df.replace([np.inf, -np.inf], np.nan, inplace=True)
#         df.dropna(inplace=True)
#         mean_df = df.groupby('subject').mean().reset_index()

         
#         ll_per_type = np.concatenate([(mean_df['neural_%s' % t]/mean_df['non-neural_%s' % t]).values[np.newaxis,:] for t in types])

#         ax[i_area,i_set].plot(
#             ll_per_type,
#             ls='-',color='grey',alpha=.7,lw=1
#             )
#         ax[i_area,i_set].plot(
#             np.median(ll_per_type,axis=1),
#             ls='-',color='r',alpha=1,lw=2
#             )
        
#         ax[i_area,i_set].axhline(1,color='k',linestyle='--')
        
#         ax[i_area,i_set].set_xticks(range(0,len(types)))
#         ax[i_area,i_set].set_xticklabels(types)
#         ax[i_area,i_set].set_ylim([0,2])

#         if i_set==0:
#             ax[i_area,i_set].set_ylabel(area)
# # %%

# %%
