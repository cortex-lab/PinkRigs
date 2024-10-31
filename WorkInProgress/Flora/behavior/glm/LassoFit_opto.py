# %%
# 
from pathlib import Path 
import pandas as pd 
import numpy as np
from scipy import stats


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from predChoice_utils import fit_model,get_weights


set_name = 'uni_all_nogo'

basepath = Path(r'D:\LogRegression\opto\%s\formatted' % set_name)
savepath = basepath / 'fit_results'
savepath.mkdir(parents=False,exist_ok=True)




all_files = list(basepath.glob('*.csv'))


# %%
params,ll = zip(*[fit_opto_model(rec,gammafit=True) for rec in all_files])
_,ll_nogamma = zip(*[fit_opto_model(rec,gammafit=False) for rec in all_files])

params = pd.concat(params)
params['neglogLoss'] = ll
params['neglogLoss_ng'] = ll_nogamma

# get the distance data etc.
params['subject'] = [rec.name.split('_')[0] for rec in all_files]
params['hemisphere'] = [rec.name.split('_')[1] for rec in all_files]

cannula_pos = pd.read_csv(r'D:\opto_cannula_locations.csv')
cannula_pos['hemisphere'] = ['right' if hem==1 else 'left' for hem in cannula_pos.hemisphere]
params = pd.merge(params,cannula_pos.iloc[:,1:],on=['subject','hemisphere'],how='outer')
# calculate distance from stimulus position
mid_loc = -3950; # location of the stimulus in AP
params['distance_from_stim'] = np.abs(-params.ap+5400-mid_loc)


# %%
import matplotlib.pyplot as plt

# compare the two
fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(params.neglogLoss,params.neglogLoss_ng,'.') # tbh the improvement is negligible... 

ax.axline([-1,-1],[0,0])
# %%
# some plots

for s in stim_predictors:
    params['%s_tot' %s] = params[s]+params['%s_opto' %s]
params['bias_tot'] = params['bias'] + params['opto']


opto_var_names = [n  for n in params.columns if 'tot' in n]
control_var_names = [n.split('_')[0] for n in opto_var_names]

# fig1 parameter change
fig,ax=plt.subplots(1,len(control_var_names),
                    figsize=(12,2),sharex=True,sharey=True)

for i,(c,o) in enumerate(zip(control_var_names,opto_var_names)): 
    ax[i].plot(params[c],params[o],'o',
               markeredgecolor='k',markerfacecolor='cyan',
               markersize=7)
    
    t_stat, p_value = stats.ttest_rel(params[c], params[o])


    ax[i].axline([-4,-4],[6,6],color='k',linestyle='--')
    ax[i].axhline(0,color='k',linestyle='--')
    ax[i].axvline(0,color='k',linestyle='--')
    ax[i].set_title('%s, %.3f' % (c,p_value))
    ax[i].set_xlabel(c)
    ax[i].set_ylabel(o)


# parameter changes compared to disntace/eYFP inensity

# %%
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm


fig,ax=plt.subplots(1,len(control_var_names),
                    figsize=(7,2.5),sharex=True)

plot_type = 'distance_from_stim'
plot_type = 'eYFP_fluorescence'


for i,(c,o) in enumerate(zip(control_var_names,opto_var_names)):
    ctrl_,opto_ = params[c],params[o]

    dpar = opto_/ctrl_ if c!='bias' else opto_-ctrl_
    nochangeline = 1 if c!='bias' else 0
    label ='opto/ctrl' if c!='bias' else 'opto-ctrl'
    minl = -.1 if c!='bias' else -4.4
    maxl = 2.1 if c!='bias' else 5

    dpar_name = 'delta_%s' % c
    params[dpar_name] = dpar

    md = mixedlm("%s ~ %s" % (dpar_name,plot_type),
             params_,groups=params_['subject'])
    result = md.fit() 
    result.summary()
    pval = result.pvalues[plot_type]
        
    ax[i].plot(params[plot_type],dpar,'o',
               markeredgecolor='k',markerfacecolor='cyan',
               markersize=7)
    
    ax[i].axhline(nochangeline,color='k',linestyle='--')
    ax[i].set_xlabel(plot_type)
    #ax[i].set_ylabel(label)
    ax[i].set_ylim([minl,maxl])
    #ax[i].set_yticklabels([])


    
savename = savepath / ('paramchange_%s.svg' % plot_type)
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
#  save the data 

params.to_csv(savepath / 'summary.csv')

# %%
