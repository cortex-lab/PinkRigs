# %%
# this code is to plot the neur psychometric per each region 
# first we obrain predictions from each trial form the neural predictors 
# then we get all trials and refit the psychometric with left vs right for all trials - maybe all mice or per mice


# generic libs 
import numpy as np
import pandas as pd
from itertools import product


from pathlib import Path 

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns


from plot_utils import plot_psychometric,copy_svg_to_clipboard
from predChoice_utils import fit_stim_vs_neur_models,get_LogOdds

# my own util funcions
# data read in

# helper function 

def fit_predict_neur(rec,neur_model='neur'):
    """
    this function fits the neurometric model and spits out what the neurons would predict next to the trials

    Parameters:
    ----------
    rec: pathlib.Path 
        pointer to to file for trials

    neur_model: str, 'neur' or 'all'
        which model to use to predict the neural choices 

    """
    trials = pd.read_csv(rec)
    results = fit_stim_vs_neur_models(trials,neuron_selector_type='lasso')


    model = results['models'][neur_model]
    X =  pd.concat((
        results['X_train_subsets'][neur_model], 
        results['X_test_subsets'][neur_model]
    ))

    neur_proba = get_LogOdds(model,X,which_features='neur')
    neur_prediction = (neur_proba>0).astype('float') 
    # we will concatenate all this to long-form

    trials['neural_prediction'] = neur_prediction

    # already do the gamma transform here
    gamma = results['params']['stim']['hyperparameters']['feature_selector__vis__power']

    for v in (['visR','visL']):
        trials[f'{v}_gamma'] = trials[v] ** gamma


    # save out a bunch of stuff in long form

    # gamma
    trials['gamma'] = gamma
    trials[['subject','expDate','expNum']] = rec.stem.split('_')


    # return the subsetted matrix (crucial to subset as different sessions have differnt neurons)
    return trials[['visL','visR','audL','audR','choice',
                      'visL_gamma','visR_gamma',
                      'neural_prediction','bias','gamma',
                      'subject','expDate','expNum']]

def plot_left_right_neural(trials,axs,plot_type='log'):

    assert len(axs)==2,(
        'this helper requires to subplots inputted'
    )
    axL,axR = axs[0],axs[1]

    nTrials_per_subject = trials.groupby('subject').size()
    trials_ = trials.groupby('subject').sample(n=min(nTrials_per_subject))

    gamma_  = trials.gamma.mean()
    plot_psychometric(trials_,gamma=gamma_,yscale=plot_type, 
                    dataplotkwargs={'marker':'','ls':''},
                    predplotkwargs={'ls':'--'},ax = axL)


    plot_psychometric(trials_[trials_.neural_prediction==0],gamma=gamma_,yscale=plot_type, 
                    dataplotkwargs={'marker':'o','ls':''},
                    predplotkwargs={'ls':'-'},ax = axL)


    plot_psychometric(trials_,gamma=gamma_,yscale=plot_type, 
                    dataplotkwargs={'marker':'','ls':''},
                    predplotkwargs={'ls':'--'},ax = axR)


    plot_psychometric(trials_[trials_.neural_prediction==1],gamma=gamma_,yscale=plot_type, 
                    dataplotkwargs={'marker':'o','ls':''},
                    predplotkwargs={'ls':'-'},ax = axR)


brain_areas = ['SCs']
paramsets = ['choice']

neur_data_sets = list(product(brain_areas,paramsets))

fig,ax = plt.subplots(len(neur_data_sets),2,
                      figsize=(8,4*len(neur_data_sets)), 
                      sharex=True,sharey=True
)


for idx,set_name in enumerate(neur_data_sets):
    region,tt = set_name[0],set_name[1]
    savepath = Path(r'D:\LogRegression\%s_%s\formatted' % (region,tt))
    all_files = list(savepath.glob('*.csv'))
    trials = [fit_predict_neur(rec,neur_model = 'all') for rec in all_files]
    trials = pd.concat(trials)
    #trials = trials[np.isin(trials.subject,['AV005', 'AV008', 'AV014', 'FT030', 'FT032', 'FT035'])]
    plot_left_right_neural(trials,ax[:],plot_type = 'log')

    ax[0].set_title(region)

plt.show()
copy_svg_to_clipboard(fig)


# %%
fig,ax = plt.subplots(len(neur_data_sets),2,
                      figsize=(8,4*len(neur_data_sets)), 
                      sharex=True,sharey=True
)

plot_left_right_neural(trials[np.isin(trials.subject,['AV005', 'AV008', 'AV014', 'FT030', 'FT032', 'FT035'])],ax[:],plot_type = 'log')

# %%
