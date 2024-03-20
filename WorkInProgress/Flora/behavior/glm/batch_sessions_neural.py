# %%
import sys 
import numpy as np
from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params

recordings = load_rec_df(recompute=False)
timing_params = load_params(paramset='choice')


# %%


model_types = ['non-neural','neural']

logLiks = np.zeros((len(model_types),len(recordings)))
b,a,v,coh,conf = logLiks.copy(),logLiks.copy(),logLiks.copy(),logLiks.copy(),logLiks.copy(),
n_neurons = []
ll_nobias = []
for r,(_,rec) in enumerate(recordings.iterrows()):

    trials,gIDs,_,_ = search_for_neural_predictors(rec,my_ROI='SCm',ll_thr = 0.01,exclude_premature_wheel=True,**timing_params)
    for i,model in enumerate(model_types):
        if model=='non-neural':
            trial_matrix = trials.iloc[:,:3]
        elif model=='neural': 
            trial_matrix = trials
            n_neurons.append(gIDs.size)

        # model 
        glm = glmFit(trial_matrix,model_type='AVSplit',
                     fixed_parameters = [0,0,0,0,0,0])        
        glm.fitCV(n_splits=2,test_size=0.5)

        logLiks[i,r] = glm.model.LogLik

        # get logLikelihood also per condition
        b[i,r],a[i,r],v[i,r],coh[i,r],conf[i,r] = glm.model.get_ll_per_condition(glm.model.LogLik_per_condition)


# %%
# %%
import matplotlib.pyplot as plt
from pathlib import Path

ll = logLiks[:,np.sum(np.isinf(logLiks) ,axis=0)==0]
normll = ll/ll[0,:]
plt.plot(ll[:2,:],ls='-',color='grey',alpha=.7,lw=1)
plt.plot(np.nanmean(ll[:2,:],axis=1),color='k',lw=2)
#plt.ylim([-0.1,2])
plt.ylabel('-Log2Likelihood')
#plt.yscale("log")   

which_figure = paramset
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = 'encoding_choice' + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#%%
from scipy import stats

stats.ttest_rel(ll[0,np.isnan(ll).sum(axis=0)==0],ll[1,np.isnan(ll).sum(axis=0)==0])

#%%
plt.plot(n_neurons,(logLiks[0]-logLiks[1]),'.',markersize=25,color='cyan',markeredgecolor='k')
plt.xlabel('#SC neurons available')
plt.ylabel('delta(LogLik)')
# %%
plt.hist(logLiks[0]-logLiks[1],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='k',histtype='step')
plt.hist(logLiks[0]-logLiks[2],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='g',histtype='step')
plt.hist(logLiks[1]-logLiks[2],lw=4,range=(-.1,1),bins=40,density=True,cumulative=False,color='r',histtype='step')
# %%
