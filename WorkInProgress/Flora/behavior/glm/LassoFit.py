# %%
# generic libs 
import numpy as np
import pandas as pd


from pathlib import Path 

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns


from predChoice_utils import make_fake_data,batch_fit_neurometric

# my own util funcions
# data read in

from itertools import product

brain_areas = ['VISp']
paramsets = ['choice']

for x in product(brain_areas,paramsets):
    region,tt = x[0],x[1]
    savepath = Path(r'D:\LogRegression\%s_%s\formatted' % (region,tt))
    all_files = list(savepath.glob('*.csv'))
    df = batch_fit_neurometric(all_files,'lasso',savepath=savepath)


# %%




# %%

import seaborn as sns 


# Plot 1: look at the degreee of overfitting

metric = 'log_loss'
models_to_compare = ['bias','stim','neur','all']


df_long = pd.melt(
    df, 
    value_vars = (
      [f'{metric}_train_{model}' for model in models_to_compare] + 
      [f'{metric}_test_{model}' for model in models_to_compare]  
    ), 
    var_name = 'model', 
    value_name = metric
)

# Step 2: Create a new column that labels each model
df_long['type'] = df_long['model'].apply(lambda x: 'test' if 'test' in x else 'train')
df_long['model'] = df_long['model'].apply(lambda x: x.split('_')[-1])



sns.boxenplot(data=df_long,
            x = 'model',
            y= metric,hue='type')

#%%
# plot 2 all x axis -- trialtype, y axis=performance, hue=model 
import itertools
metric = 'roc_auc_score'
trial_types = ['blank','vis','aud','coherent','conflict']
models_to_compare = ['all']

for condition in trial_types:
    #df[f'{condition}_roc_auc_score_neur'] -= df[f'{condition}_roc_auc_score_stim']
    df[f'{condition}_roc_auc_score_all'] -= df[f'{condition}_roc_auc_score_stim']

#%%

df_long = pd.melt(
    df, 
    value_vars = (
        [f'{trial_type}_{metric}_{model}' 
        for trial_type,model in itertools.product(trial_types,models_to_compare)]
    ), 
    var_name = 'long_name', 
    value_name = metric
)
df_long[['trial_type','model']] = df_long['long_name'].str.split('_',expand=True).iloc[:,[0,-1]]
df_long = df_long.drop(columns='long_name')

#df_long = df_long.dropna()

plt.rcParams.update({'font.size': 18})

fig,ax = plt.subplots(1,1,figsize=(3.7,2.6))
sns.lineplot(data = df_long,
            x = 'trial_type', y = metric, hue = 'model',
            palette= {'all':'green','neur':'orange'}
)
ax.set_ylim([-0.1,0.3])
ax.axhline(0,color='k',linestyle='--')
plt.legend().remove()
plt.xticks(ticks=[0, 1, 2, 3,4], 
           labels=['ø','V', 'A', 'V=A', 'V≠A'],
           rotation=0)
plt.xlabel('')
plt.tight_layout()
plt.title(brain_areas[0])
# plt.ylabel('')
# plt.yticks([])

from plot_utils import copy_svg_to_clipboard,off_axes
off_axes(ax,'top')
plt.show()
copy_svg_to_clipboard(fig)
# %%
#import pandas as pd

#df = pd.read_csv(r'D:\LogRegression\MOs_poststim\result_sfs_2024-09-05-113908.csv')
#   #%%
# kappas = np.concatenate(kappas,axis=1)
# betas = np.concatenate(betas,axis=1)

# # %%
# fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
# ax[0].plot(kappas[0,:],kappas[1,:],'x',color='magenta')
# ax[0].axline([0,0],[2,2],color='k')
# ax[0].set_title('kappas')
# ax[0].set_xlabel('neg bin')
# ax[0].set_ylabel('pos bin')



# ax[1].plot(betas[0,:],betas[1,:],'x',color='magenta')
# ax[1].axline([-2,-2],[2,2],color='k')
# ax[1].set_title('betas')
# ax[1].set_xlabel('neg bin')
# ax[1].set_ylabel('pos bin')

#%%
import matplotlib.pyplot as plt
plt.plot(pd.concat(logLosses).T,color='grey')
plt.plot(pd.concat(logLosses).mean(),color='red')
plt.axhline(0,color='k',linestyle='--')
plt.ylabel('neglogLoss,all/neglogLoss,stim')
# %%
import seaborn as sns
sns.relplot(data=df,x='negLL_stim',y='negLL_all',hue='subject',size='|initBias|')
plt.axline((0,0),slope=1,color='k',linestyle='--')


# %%
wh = '|initBias|'
plt.plot(df[wh],df.negLL_all -df.negLL_stim,'o')
plt.xlabel(wh)

plt.ylabel('LogLoss,all - LogLoss,stim')
# %%
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm


df_long = pd.melt(df, id_vars=['subject'], value_vars=['negLL_all', 'negLL_stim'], var_name='condition', value_name='negLL')

# Display the reshaped DataFrame
print(df_long.head())
model = mixedlm("negLL ~ condition", df_long, groups=df_long["subject"])
result = model.fit()

# Print the summary of the model fit
print(result.summary())

# %%


# %%
from predChoice_utils import Odds_hists


models = {'negative': results['models']['all'],'positive':results['models']['all']}
Xs = {'negative':results['X_tests']['all'],'positive':results['X_tests']['all']}

oddkwargs = [{
'which_features':'binned_neur_weights',
'weight_min':-100,'weight_max':0
},{
'which_features':'binned_neur_weights',
'weight_min':0,'weight_max':100
}]

Odds_hists(models,Xs,oddkwargs=oddkwargs)

# %%
