# %%

# generic libs 
import numpy as np
import pandas as pd
from pathlib import Path 
import seaborn as sns 

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns


from predChoice_utils import fit_stim_vs_neur_models

# my own util funcions
# data read in
savepath = Path(r'D:\LogRegression\SCm_prestim')
all_files = list(savepath.glob('*.csv'))

logLosses,llall,llstim,subjects,nTrials,nNeur,nNeur_used,pCorrect,initBias,kappas,betas = [],[], [],[],[],[],[],[],[],[],[]
for idx,rec in enumerate(all_files):
    trials = pd.read_csv(rec)

    print(rec.name)

    results  = fit_stim_vs_neur_models(trials,
                            plot_odds=False,
                            plot_odds_neural = True,
                            plot_AUCs=False,
                            plot_neur_weights=False,
                            plot_stim_weights=False)
    

    stim_neur_logLoss = results['test_scores_per_trial_type']['all'].values-results['test_scores_per_trial_type']['stim'].values

    stim_neur_logLoss = pd.DataFrame(stim_neur_logLoss,
                                    columns=results['test_scores_per_trial_type']['stim'].columns.values)
    
    logLosses.append(stim_neur_logLoss)

    subjects.append(rec.name.split('_')[0])
    llall.append(results['test_scores']['all'])
    llstim.append(results['test_scores']['stim'])
    nTrials.append(results['X_tests']['all'].shape[0])
    nNeur.append(results['X_tests']['all'].shape[1]-4)
    nNeur_used.append(np.sum(results['params']['all']['weights'].values!=0)-4)
    pCorrect.append(np.mean(trials.feedback==1))
    initBias.append(results['params']['stim']['bias'])
    kappas.append(np.array(results['kappas']).reshape(-1,1))
    betas.append(np.array(results['betas']).reshape(-1,1))
#%%
kappas = np.concatenate(kappas,axis=1)
betas = np.concatenate(betas,axis=1)

# %%
fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax[0].plot(kappas[0,:],kappas[1,:],'x',color='magenta')
ax[0].axline([0,0],[2,2],color='k')
ax[0].set_title('kappas')
ax[0].set_xlabel('neg bin')
ax[0].set_ylabel('pos bin')



ax[1].plot(betas[0,:],betas[1,:],'x',color='magenta')
ax[1].axline([-2,-2],[2,2],color='k')
ax[1].set_title('betas')
ax[1].set_xlabel('neg bin')
ax[1].set_ylabel('pos bin')

#%%
plt.plot(pd.concat(logLosses).T,color='grey')
plt.plot(pd.concat(logLosses).mean(),color='red')
plt.axhline(0,color='k',linestyle='--')
plt.ylabel('neglogLoss,all/neglogLoss,stim')
# %%
# all the recordings and mice etc.

df = pd.DataFrame({
    'subject':subjects, 
    'negLL_all':llall,
    'negLL_stim':llstim,
    'nTrials': nTrials,
    'nNeur': nNeur,
    'nNeur_used':nNeur_used,
    'pCorrect':pCorrect,
    'pIncorrect': 1-np.array(pCorrect),
    'nIncorrect':np.array(nTrials)*(1-np.array(pCorrect)),
    '|initBias|':np.abs(initBias),

})

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
