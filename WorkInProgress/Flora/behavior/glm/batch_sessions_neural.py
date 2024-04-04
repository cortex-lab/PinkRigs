# %%
 
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats

from predChoice import glmFit,search_for_neural_predictors
from loaders import load_rec_df,load_params

brain_area = 'RSPv'     
  
paramsets = ['choice']
#paramset ='choice' 

for paramset in paramsets:
    recordings = load_rec_df(brain_area=brain_area,recompute_session_selection=True,paramset=paramset)
    savepath= Path(r'D:\ChoiceEncoding\LogLikeilihoods_%s_%s.csv' % (brain_area,paramset))


    timing_params = load_params(paramset=paramset)

    model_types = ['non-neural','neural']

    logLiks = np.zeros((len(model_types),len(recordings)))
    b,a,v,coh,conf = logLiks.copy(),logLiks.copy(),logLiks.copy(),logLiks.copy(),logLiks.copy(),
    n_neurons = []
    ll_nobias = []
    for r,(_,rec) in enumerate(recordings.iterrows()):

        trials,gIDs,_,_ = search_for_neural_predictors(rec,my_ROI=brain_area,ll_thr = 0.01,exclude_premature_wheel=False,**timing_params)
        for i,model in enumerate(model_types):
            if model=='non-neural':
                trial_matrix = trials.iloc[:,:3]
            elif model=='neural': 
                trial_matrix = trials
                n_neurons.append(gIDs.size)

            # model 
            glm = glmFit(trial_matrix,model_type='AVSplit',
                        fixed_parameters = [0,0,0,0,1,0])        
            glm.fitCV(n_splits=2,test_size=0.5)
            print(r,glm.model.LogLik)
            logLiks[i,r] = glm.model.LogLik

            # get logLikelihood also per condition
            b[i,r],a[i,r],v[i,r],coh[i,r],conf[i,r] = glm.model.get_ll_per_condition(glm.model.LogLik_per_condition)


    # 
            
    df = pd.DataFrame({
                    'non-neural_total':logLiks[0,:],
                    'neural_total':logLiks[1,:],
                    'non-neural_blank':b[0,:],
                    'neural_blank':b[1,:],
                    'non-neural_aud':a[0,:],
                    'neural_aud':a[1,:],    
                    'non-neural_vis':v[0,:],
                    'neural_vis':v[1,:], 
                    'non-neural_coh':coh[0,:],
                    'neural_coh':coh[1,:],     
                    'non-neural_conf':conf[0,:],
                    'neural_conf':coh[1,:],                 
                    'subject':recordings.subject.values,
                    'expDate':recordings.expDate.values,
                    'expNum': recordings.expNum.values
                    })
    
    # save the logLikelihood results
    df.to_csv(savepath)

    # drop rows with Inf and nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # average across subjects
    mean_df = df.groupby('subject').mean().reset_index()

    _,p_value = stats.ttest_rel(mean_df['non-neural_total'], mean_df['neural_total'])



    plt.plot(mean_df[['non-neural_total','neural_total']].values.T,ls='-',color='grey',alpha=.7,lw=1)
    plt.plot(mean_df[['non-neural_total','neural_total']].values.T.mean(axis=1),ls='-',color='k',alpha=1,lw=2)
    #plt.ylim([-0.1,2])
    plt.ylabel('-Log2Likelihood')
    plt.title('p = %.4f' % p_value)

    which_figure = paramset
    cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
    im_name = 'encoding_choice' + which_figure + '.svg'
    savename = cpath / im_name #'outline_brain.svg'
    plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
