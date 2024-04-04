
#%%
import pandas as pd
from pathlib import Path

from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params
brain_area = 'SCm'
paramset = 'choice'

recordings = load_rec_df(brain_area=brain_area,paramset=paramset,recompute_session_selection=False)

timing_params = load_params(paramset=paramset)


#

rec = recordings.iloc[11]

trials,gIDs,best_nrn,ll_best = search_for_neural_predictors(rec,my_ROI=brain_area,ll_thr = 0.01,exclude_premature_wheel=False,**timing_params)



non_neural = trials.iloc[:,:3]
neural = trials.iloc[:,3:]
glm = glmFit(non_neural,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = [1,1,1,1,.7,0])
glm.fitCV(n_splits=2,test_size=0.5)



# look at the metrics
import matplotlib.pyplot as plt
# refit and assess model contribution of each parameter
fig,ax = plt.subplots(1,1,figsize=(8,8))
final_matrix = pd.concat((non_neural,neural.loc[:,best_nrn]),axis=1)
final_glm = glmFit(final_matrix,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = list(glm.model.allParams))   
final_glm.fitCV(n_splits=2,test_size=0.5)
final_glm.visualise(yscale='log',ax=ax)
fig.suptitle('{subject}_{expDate}_{expNum}'.format(**rec))
ax.set_title('LogLik=%.2f' % final_glm.model.LogLik)

# compare the neural vs non-neural models

#
fig,ax = plt.subplots(1,1,figsize=(8,8))

final_glm.plotPrediction(yscale='log',ax=ax)
ax.axline((0,0),slope=1,color='k',linestyle='--')
ax.set_xlabel('actual')
ax.set_ylabel('predicted')
ax.set_title('LogOdds')
#
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(ll_best)
ax.set_xlabel('# best neuron')
ax.set_ylabel('-Log2Likelihood')
ax.axhline(final_glm.model.LogLik,color='k',ls='--')
ax.text((len(ll_best)-5),final_glm.model.LogLik+0.004,'refit with all %.0f neurons' % (len(ll_best)-1))
ax.set_title('improvement on prediction from SC neuron prior to %s' % paramset)


which_figure = 'LogOdds_neural_improvement_per_neurons'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


fig,ax = plt.subplots(2,1,figsize=(25,5))
gParams = list(final_glm.model.required_parameters.copy())
gParams = gParams + best_nrn
ax[0].plot(gParams,final_glm.model.allParams,'o-')
ax[1].axhline(0,color='k',ls='--')
ax[1].set_xlabel('neuronID') 

which_figure = 'LogOdds_neural_parameters'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#%%
import numpy as np
fig,ax = plt.subplots(2,3,figsize=(15,8))
for i in range(2):

    if i==0:
        mytitle ='non-neural'
        m = non_neural
    elif i==1: 

        mytitle ='with neurons'
        m = final_matrix

    testglm = glmFit(m,model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0],fixed_paramValues = list(glm.model.allParams))
    testglm.fitCV(n_splits=2,test_size=0.5)
    testglm.visualise(yscale='log',ax = ax[0,i])
    ax[0,i].set_title('%s, LogLik: %.2f' % (mytitle,testglm.model.LogLik))
    testglm.plotPrediction(yscale='log',ax=ax[1,i])
    ax[1,2].hist(np.abs(testglm.model.get_logOdds(testglm.conditions,testglm.model.allParams))
                 ,alpha=0.5,density=False,range=(0,10)) 
    ax[1,1].set_ylim([-5.5,5.5])
    ax[1,0].set_ylim([-5.5,5.5])
    ax[1,1].set_xlim([-5.5,5.5])
    ax[1,0].set_xlim([-5.5,5.5])

ax[1,2].set_xlabel('LogOdds')
ax[1,2].set_ylabel('# trials')
ax[1,2].legend(['non-neural','neural'])

from Analysis.pyutils.plotting import off_axes
off_axes(ax[0,2])

which_figure = 'LogOdds_neural_improvement_examlple'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
