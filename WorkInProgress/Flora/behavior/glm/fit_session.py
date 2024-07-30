
#%%
import pandas as pd
from pathlib import Path

from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params
brain_area = 'MRN'
paramset = 'choice'

recordings = load_rec_df(brain_area=brain_area,paramset=paramset,recompute_session_selection=False)

timing_params = load_params(paramset=paramset)


#
# %%
rec = recordings.iloc[4]

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
final_glm.visualise(yscale='sig',ax=ax)
fig.suptitle('{subject}_{expDate}_{expNum}'.format(**rec))
ax.set_title('LogLik=%.2f' % final_glm.model.LogLik)

# compare the neural vs non-neural models




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

# logOdds neural
logOdds_per_trial_stim = glm.model.get_logOdds(glm.conditions,
                                               glm.model.allParams)
logOdds_per_trial_neural = final_glm.model.get_logOdds(final_glm.conditions,
                                                       final_glm.model.allParams)
audDiff = glm.conditions[:,0]
visDiff = glm.conditions[:,1]
choices = glm.choices
trial_types = glm.model.get_trial_types(visDiff,audDiff)



# plot it with seaborn 
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

df = pd.DataFrame({
    'LogOdds,stim': logOdds_per_trial_stim,
    'LogOdds,neural': logOdds_per_trial_neural,
    'choice':choices,
    'trial_types':trial_types
})

jointdat = np.array([logOdds_per_trial_stim,logOdds_per_trial_neural])

g= sns.jointplot(data=df,
              x="LogOdds,stim", y="LogOdds,neural", hue='choice',
              style = trial_types, 
              style_order = ['blank','vis','aud','coherent','conflict'],
              markers = ['o','v','^','P','X'],size=np.abs(visDiff)+1,
              palette='coolwarm',hue_norm=(.1,.9),
              kind='scatter',ratio=4,edgecolor='k',
              xlim=[np.min(jointdat)*1.05,np.max(jointdat)*1.05],
              ylim=[np.min(jointdat)*1.05,np.max(jointdat)*1.05],
              legend=False)


g.ax_joint.axline((0,0),slope=1,color='k',linestyle='--')
g.ax_joint.axvline(0,color='k',linestyle='--')
g.ax_joint.axhline(0,color='k',linestyle='--')
g.ax_marg_x.set_title('AUC = %.2f' % 
                      roc_auc_score(df['choice'].values,logOdds_per_trial_stim))

g.ax_marg_y.set_title('AUC = %.2f' % 
                      roc_auc_score(df['choice'].values,logOdds_per_trial_neural))
g.ax_marg_x.legend(['right choice','left choice'])

plt.show()

# %%
# now let's see Matteo's plot -- i.e. Roy et al., 2018 

# log-Odds range
def raise_to_sigm(logOdds):
    return np.exp(logOdds) / (1 + np.exp(logOdds))

def get_pR_per_bin(actual_odds,bin_pos):
    """
    function that first bins the odds values to equidistant bins 
    and then gets the average pR within each odds bin 

    """
    indices = np.digitize(actual_odds,bins=bin_pos)
    pR_per_trial = raise_to_sigm(actual_odds)
    # average across bins 
    pR_per_bin = [np.mean(pR_per_trial[indices==i]) for i in range(bin_pos.size)]
    
    return np.array(pR_per_bin)

minOdd = np.min(jointdat)*1.05
maxOdd = np.max(jointdat)*1.05

fig,(ax,axh) = plt.subplots(2,1,figsize=(6,10),
                            sharex=True,
                            gridspec_kw={'height_ratios':[3,1]})

# the sigmoid curve
odds = np.linspace(minOdd,maxOdd,100)
ax.plot(odds,raise_to_sigm(odds),color='k',alpha=.5)
# th actual models predictions

color_stim = '#E1BE6A'
color_neural = '#40B0A6'

bin_pos = np.arange(minOdd,maxOdd,.7)
ax.scatter(bin_pos,get_pR_per_bin(logOdds_per_trial_stim,bin_pos),
         s=30,c=color_stim,edgecolors='k')
ax.scatter(bin_pos,get_pR_per_bin(logOdds_per_trial_neural,bin_pos),
           s=30,c=color_neural,edgecolors='k')
axh.hist(logOdds_per_trial_stim,bins=bin_pos,
         rwidth=.9,alpha=.7,align='right',color=color_stim)
axh.hist(logOdds_per_trial_neural,bins=bin_pos,
         rwidth=.9,alpha=.7,color=color_neural)
axh.set_xlabel('logOdds') 
ax.set_ylabel('pR')
ax.set_ylim([-.05,1.05])
axh.legend(['stim','neural'])

axh.axvline(0,color='k',linestyle='--')
ax.axvline(0,color='k',linestyle='--')
ax.axhline(0.5,color='k',linestyle='--')

# %%
