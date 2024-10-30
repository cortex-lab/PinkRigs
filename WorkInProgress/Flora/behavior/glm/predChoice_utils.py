
import datetime
from pathlib import Path 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import zscore
import itertools

from sklearn.metrics import roc_auc_score,log_loss,accuracy_score,balanced_accuracy_score,f1_score
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.feature_selection import SelectFromModel,VarianceThreshold,SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import warnings
# we ignore the warning that we can't everwrite part of pd.df
# and also it needs to remain possible that the neural model simply finds no good features, so we suppress that warning
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore",message="No features were selected: either the data is too noisy or the selection test too strict")

def get_stimDiffs(X):
    visDiff = X['visL']*-1 + X['visR']
    audDiff = X['audL']*-1 + X['audR']   
    return visDiff,audDiff

def get_trial_bools(visDiff,audDiff):
    """
    function to return a booleans for trial types 
    returns:
    blank, visual, auditory, cohrent and conflict as bools 
    """
    
    is_blank = (visDiff==0) & (audDiff==0)
    is_vis = (visDiff!=0) & (audDiff==0)
    is_aud = (visDiff==0) & (audDiff!=0)
    is_coh = (visDiff!=0) & (audDiff!=0) & ((np.sign(audDiff)-np.sign(visDiff))==0)
    is_conf = (visDiff!=0) & (audDiff!=0) & ((np.sign(audDiff)-np.sign(visDiff))!=0)

    return is_blank,is_vis,is_aud,is_coh,is_conf

def get_trial_types(visDiff,audDiff):
    """
    function that retunrns an np.array of strings based that will indicate what type of trial we are in (vis/aud/coh/conflict/blank)
    """
    is_blank,is_vis,is_aud,is_coh,is_conf= get_trial_bools(visDiff,audDiff)


    trial_types = np.empty(visDiff.shape, dtype='U8')
    trial_types[is_blank]=['blank']
    trial_types[is_vis]=['vis']
    trial_types[is_aud]=['aud']
    trial_types[is_coh]=['coherent']
    trial_types[is_conf]=['conflict']

    return trial_types

# model utilties
def get_stim_predictors(X):
    '''
    gets a boolean for which column in the predictor matrix is from stimuli
    '''
    is_stim = [('vis' in col_name) or ('aud' in col_name) 
     for col_name,_ in X.items()]
    
    return np.array(is_stim)


### potentially truncate 
def get_params(model,X):
    # get hyperameters if model was fitted with grid search
    if hasattr(model,'best_model'):
        gamma = np.array([model.best_params_['gamma_transformer__kw_args']['gamma']])
        model = model.best_estimator_

    selected_features = model.named_steps['feature_selector'].get_support()
    coefficients = model.named_steps['logistic_regression'].coef_
    intercept = model.named_steps['logistic_regression'].intercept_    
    params = pd.DataFrame(coefficients,columns=X.iloc[:,selected_features].columns)
    params['gamma'] = gamma
    params['bias'] = intercept
    return params

# potentially truncate
def get_LogOdds(model,X,which_features = 'all',add_bias = False,weight_min=0,weight_max=100):
    '''
    Function to get prediction of the model in terms of LogOdds, 
    with the option to get the odds predicting from only a subset of features.

    Parameters:
    -----------
    model : a fitted sklearn Pipeline object containing a feature selector and logistic regression model
    X : array-like or sparse matrix of shape (n_samples, n_features)
        The input samples.
    which_features : str, optional (default='all')
        The subset of features to use:
        'all' : use all features
        'stim' : use only non-neuron features
        'neur' : use only neuron features
        'binned_neur_weights' : use neuron features within a specific weight range (specified by weight_min and weight_max)
    add_bias: bool, optional (default=False)
        whether to add the intercept to the odds or not
    weight_min : float, optional (default=0)
        The minimum weight for features to be included when which_features='binned_neur_weights'.
    weight_max : float, optional (default=100)
        The maximum weight for features to be included when which_features='binned_neur_weights'.

    Returns:
    --------
    LogOdds : array-like of shape (n_samples,)
        The log-odds of the input samples.
    '''

    selector = model.named_steps['feature_selector']
    X_ = selector.transform(X)

    weights = model.named_steps['logistic_regression'].coef_[0] 
    intercept =  model.named_steps['logistic_regression'].intercept_[0]

    feature_names = selector.get_feature_names_out()

    if which_features=='all': 
        kept_features = feature_names
    elif which_features=='stim':
        kept_features = [k for k in feature_names if 'neuron' not in k]
    elif which_features=='neur':
        kept_features = [k for k in feature_names if 'neuron' in k] 
    elif which_features=='binned_neur_weights':
        kept_features = [k for k,w in zip(feature_names,weights) if (('neuron' in k) and ((w>weight_min)&(w<weight_max)))] 
        stim_features = [k for k in feature_names if 'neuron' not in k]
        kept_features = kept_features + stim_features


    is_kept = np.isin(feature_names,kept_features).astype('int')
    LogOdds = np.dot(X_,weights*is_kept)
    
    if add_bias: 
        LogOdds+=intercept

    return LogOdds

 

def raise_to_sigm(logOdds):
    'function to calculate pR from logOdds'
    return np.exp(logOdds) / (1 + np.exp(logOdds))

# plotting functions
def plot_Odds(logOdds,
              minOdd=None,maxOdd=None,
              ax=None,axh=None,color='#E1BE6A',label=None):
    
    # some helper functions

    # raise the odds to the sigmoid

    
    # bin the probabilities
    def get_pR_per_bin(actual_odds,bin_pos):
        """
        function that first bins the odds values to equidistant bins 
        and then gets the average pR within each odds bin 

        """
        indices = np.digitize(actual_odds,bins=bin_pos)
        pR_per_trial = raise_to_sigm(actual_odds)
        # average across bins 
        mid_bins = bin_pos[:-1]+np.diff(bin_pos)/2
        pR_per_bin = [np.nanmean(pR_per_trial[indices==i+1]) for i in range(bin_pos.size-1)]
    
        return mid_bins,np.array(pR_per_bin)

    if minOdd is None:
        minOdd = np.min(logOdds)*1.05
    if maxOdd is None:
        maxOdd = np.max(logOdds)*1.05

    if ax is None:
        _,(ax,axh) = plt.subplots(2,1,figsize=(6,10),
                                    sharex=True,
                                    gridspec_kw={'height_ratios':[3,1]})
   
   
    axh.axvline(0,color='k',linestyle='--')
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0.5,color='k',linestyle='--')

    # the sigmoid curve
    odds = np.linspace(minOdd,maxOdd,100)
    ax.plot(odds,raise_to_sigm(odds),color='k',alpha=.5)
    # th actual models predictions

    bin_pos = np.linspace(minOdd,maxOdd,8)
    mid_bins,pR_per_bin = get_pR_per_bin(logOdds,bin_pos)
    ax.scatter(mid_bins,pR_per_bin,
            s=30,edgecolors='k',c=color)

    axh.hist(logOdds,bins=bin_pos,
            rwidth=.9,alpha=.7,align='mid',color=color,label=label)
    
    # midline_kw = {
    #     'color':'k',
    #     'alpha':.2, 
    #     'linestyle':':'
    # }

    #[ax.axvline(m,ymax=v,**midline_kw) for m,v in zip(mid_bins,pR_per_bin)]
    #[axh.axvline(m,**midline_kw) for m in mid_bins]

    axh.set_xlabel('logOdds') 
    ax.set_ylabel('pR')
    ax.set_ylim([-.05,1.05])
    #axh.legend(['stim','neural'])


# potentially rewrite
def plot_psychometric(X,y,model,
                      yscale='log',ax=None,
                      colors=['b','grey','red'],
                      dataplotkwargs={'marker':'o','ls':''},
                      predpointkwargs ={'marker':'*','ls':''},
                      predplotkwargs={'ls':'-'}):
    """
    plot the model prediction for this specific model
    if the model has neural components we 0 those out
    """
    if ax is None:
        _,ax = plt.subplots(1,1,figsize=(8,8))
    
    parameters = get_params(model,X)

    # create fake stim only X matrix # quite ugly ngl
    nPredPoints = 300  
    V_model = np.concatenate((np.zeros(nPredPoints),np.linspace(0,1,nPredPoints)))
    A_model_0 = np.zeros(nPredPoints*2)
    A_model_1 = np.ones(nPredPoints*2)
    X_fake =np.concatenate((
        np.array((V_model,np.flip(V_model),A_model_0,A_model_0)).T,
        np.array((V_model,np.flip(V_model),A_model_1,A_model_0)).T,
        np.array((V_model,np.flip(V_model),A_model_0,A_model_1)).T
    ))
    neurals = np.zeros((X_fake.shape[0],X.shape[1]-4))
    X_fake = np.concatenate((X_fake,neurals),axis=1)
    X_fake = pd.DataFrame(X_fake,columns=X.columns)

    Xs = [X_fake,X,X]
    cols = [get_stim_predictors(X),None,None]
    choices = [None,None,y]
    kws = [predplotkwargs,predpointkwargs,dataplotkwargs]

    for X_,w_,plot_kws,y_ in zip(Xs,cols,kws,choices):

        if y_ is None: 
            logOdds = get_LogOdds(model,X_,columns=w_)
            pR = raise_to_sigm(logOdds)
        else:
            pR = y_

        visDiff,audDiff = get_stimDiffs(X_)
        
        Vs = np.unique(visDiff)
        As = np.unique(audDiff)

        Vmesh,Amesh =np.meshgrid(Vs,As)
        for v,a,mycolor in zip(Vmesh,Amesh,colors):
            x = v
            x = np.sign(x)*np.abs(x)**parameters.gamma.values
            y_pred = np.array([np.mean(pR[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
            if yscale=='log':
                y_pred = np.log(y_pred/(1-y_pred))
            
            ax.plot(x,y_pred,color=mycolor,**plot_kws)


    if yscale=='log':
        ax.axhline(0,color='k',ls='--')
        ax.set_ylabel('log(pR/(1-pR))')
    else:
        ax.axhline(.5,color='k',ls='--')
        ax.set_ylabel('pR')

    ax.axvline(0,color='k',ls='--')
    ax.set_xlabel('signed contrast')

# model comparison plots
def plot_AUCs(logOdds_stim,logOdds_neur,visDiff,audDiff,choices):
    '''
    this function plots the stim only vs all the predictors odds
    '''
    trial_types = get_trial_types(visDiff,audDiff)

    df = pd.DataFrame({
        'LogOdds,stim': logOdds_stim,
        'LogOdds,neural': logOdds_neur,
        'choice':choices,
        'trial_types':trial_types
    })

    jointdat = np.array([logOdds_stim,logOdds_neur])

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
                        roc_auc_score(df['choice'].values,logOdds_stim))

    g.ax_marg_y.set_title('AUC = %.2f' % 
                        roc_auc_score(df['choice'].values,logOdds_neur))
    g.ax_marg_x.legend(['right choice','left choice'])

    plt.show()

    return g 

def Odds_hists(models,Xs,cmap='viridis',oddkwargs=None,labels=None):
    '''
    Function to plot histograms of the log-odds for multiple models and datasets.
    
    Parameters:
    -----------
    models : dict/list 
        A dictionary of fitted sklearn Pipeline objects containing a feature selector and logistic regression model.
    Xs : dict/list
        A dictionary of datasets to be used for the predictions.
    cmap : str, optional (default='viridis')
        The colormap to use for the histograms.
    oddkwargs : list of dicts, optional (default=None)
        A list of dictionaries, where each dictionary contains keyword arguments for the get_LogOdds function.
    
    Returns:
    --------
    None
    '''


    # Convert models and datasets to lists
    if type(models) is not list: 
        models_ = list(models.values())
    else:
        models_ = models

    if type(Xs) is not list:    
        Xs_ = list(Xs.values())
    else: 
        Xs_ = Xs

    # Handle oddkwargs
    if oddkwargs is None:
        oddkwargs_ = [{} for _ in Xs_]  # Use empty dictionaries if oddkwargs is None
    else:
        oddkwargs_ = oddkwargs
    
    if labels is None:
        labels_ = [None for _ in Xs_]  # Use empty dictionaries if oddkwargs is None
    else:
        labels_ = labels

    # Ensure the lengths of the lists match
    assert len(oddkwargs_)==len(Xs_)==len(models_),'ambiguous request of odds'

    # Calculate LogOdds for each model and dataset
    LogOdds = [get_LogOdds(model,X,**okw) for model,X,okw in zip(models_,Xs_,oddkwargs_)]



    # Create subplots
    _,axs = plt.subplots(2,1,figsize=(6,10),
                                sharex=True,
                                gridspec_kw={'height_ratios':[3,1]})
    

    # match plotting parameters across plots by a unified dict
    odd_plot_kws ={
        'minOdd':min(np.concatenate((LogOdds))),
        'maxOdd':max(np.concatenate((LogOdds))),
        'ax':axs[0],
        'axh':axs[1]
    }
    colormap = plt.get_cmap(cmap)

    # Use the colormap
    colors = colormap(np.linspace(.2, .8, len(LogOdds)))

    [plot_Odds(l_o,color=my_color,label=my_label,**odd_plot_kws) for l_o,my_color,my_label in zip(LogOdds,colors,labels_)]

    return axs

def AUCs_compare(models,Xs,y):
    '''
    2D plot to compare two model's performances using the AUCs

    Parameters:
    ----------- 

    '''

    models_ = list(models.values())
    Xs_ = list(Xs.values())



    LogOdds = [model.decision_function(X) for model,X in zip(models_,Xs_)]
    model1_Odds,model2_Odds =LogOdds[0],LogOdds[1]
    
    visDiff,audDiff = get_stimDiffs(Xs_[0])

    fig = plot_AUCs(
        model1_Odds,model2_Odds,
        visDiff,audDiff,y
        )

    return fig

def get_scores_per_trialType(model,X,y):
    visDiff,audDiff = get_stimDiffs(X)
    trialTypes = get_trial_types(visDiff,audDiff)
    unique_trials = np.unique(trialTypes)
    scores = {t:[model.score(X[trialTypes==t],y[trialTypes==t])] for t in unique_trials}
    return pd.DataFrame(scores)


def plot_neural_psychometric(model,X,y,cmap='coolwarm'):
    """
    
    """
    
    predStim = get_LogOdds(model,X,which_features = 'stim',add_bias=True)

    predNeur = get_LogOdds(model,X,which_features = 'neur',add_bias=False)

    if y is not np.ndarray: 
        y=np.array(y)

    fig,ax = plt.subplots(2,2,figsize=(10,10))

    ax[0,0].plot(predStim,label='stim',color='magenta')
    ax[0,0].plot(predNeur,label='neur',color='cyan')
    ax[0,0].plot(y,color='k',marker='o',label='choice')
    ax[0,0].axhline(0,color='grey',linestyle='--')
    ax[0,0].set_xlabel('trial number')
    ax[0,0].set_ylabel('LogOdds')
    ax[0,0].legend()

    ax[1,0].plot(predStim,predNeur,'.')
    ax[1,0].set_xlabel('LogOdds,stim')
    ax[1,0].set_ylabel('LogOdds,neur')


    from sklearn.linear_model import LogisticRegression
    bin_pos_stim = np.linspace(min(predStim),max(predStim),8)
    bin_pos_stim_pred = np.linspace(min(predStim),max(predStim),100)

    indices_stim = np.digitize(predStim,bin_pos_stim)
    mid_bins = bin_pos_stim[:-1]+np.diff(bin_pos_stim)/2

    # neur bins
    # bin_pos_neur = np.array([min(predNeur),0,max(predNeur)*1.05])
    # indices_neur = np.digitize(predNeur,bins=bin_pos_neur)

    n_splits = 2
    indices_neur = np.array_split(np.argsort(predNeur),n_splits)
    colormap = plt.get_cmap(cmap)

    colors = colormap(np.linspace(.1, .9, n_splits))
    #colors = ['blue','red']

    kappas,betas = [],[]
    for n,is_current in enumerate(indices_neur):
        #is_current = indices_neur==idx
        choices = y[is_current]
        
        stimuli = predStim[is_current]
        # fit simple Logistic regression again
        pR_per_bin = [np.nanmean(choices[indices_stim[is_current]==i+1]) for i in range(bin_pos_stim.size-1)]

        ax[0,1].plot(mid_bins,pR_per_bin,'o',color=colors[n])

        m = LogisticRegression()
        m.fit(stimuli.reshape(-1,1),choices)
        prediction = m.predict_proba(bin_pos_stim_pred.reshape(-1,1))[:,1]
        ax[0,1].plot(bin_pos_stim_pred,prediction,color=colors[n])

        ax[0,1].set_ylim([-.1,1.1])
        ax[0,1].set_xlabel('LogOdds,stim')
        ax[0,1].set_ylabel('pR')

        ax[1,1].hist(predNeur[is_current],color=colors[n])
        ax[1,1].set_xlabel('LogOdds,neur')
        
        kappas.append(m.coef_[0][0])
        betas.append(m.intercept_[0])

    return kappas,betas
    
class PowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, power=2):
        self.power = power

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        return np.power(X, self.power)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else input_features


def fit_model(X,y,power=1,gridCV_vis=False,gridCV_neur=False, neuron_selector='lasso'):

    if neuron_selector=='lasso':
        neural_transformer = Pipeline([
            ('variance_thr',VarianceThreshold(threshold=0.001)),
            ('lasso',SelectFromModel(LogisticRegression(
                penalty='l1',solver='liblinear'),threshold=0.2))
            ])
    elif neuron_selector=='sfs':

        neural_transformer = Pipeline([
            ('variance_thr',VarianceThreshold(threshold=0.001)),
            ('sfs', SequentialFeatureSelector(
                    estimator=LogisticRegression(),
                    n_features_to_select='auto',
                    tol=  0.01,  
                    direction='forward'))  # 'forward' or 'backward') 
            ])        

    neural_predictors = [c for c,_ in X.items() if 'neuron' in c]
    is_neural_predictor = np.isin(X.columns,neural_predictors)
    is_vis_predictor = np.isin(X.columns,['visL','visR','visR_opto', 'visL_opto'])
    
    combined_transformer = ColumnTransformer(
        [
            ('neural',neural_transformer,is_neural_predictor),
            ('vis',PowerTransformer(power=power),is_vis_predictor)
         ],
        remainder='passthrough',
        force_int_remainder_cols = False,
        verbose_feature_names_out= False
    )

    

    pipeline = Pipeline([
        ('feature_selector',combined_transformer),
        ('logistic_regression',LogisticRegression(fit_intercept=False))
    ])

    if (not gridCV_vis) and (not gridCV_neur):

        pipeline.fit(X,y)    
        return pipeline
    
    else:
        
        param_grid = {}
        if gridCV_vis:
            param_grid['feature_selector__vis__power'] = np.round(np.arange(0.1,2,0.1),2)
        
        if gridCV_neur:
            if neuron_selector=='lasso':
                param_grid['feature_selector__neural__lasso__threshold'] = [0.01,0.05,0.1,0.2,0.3,0.5,1]
            if neuron_selector=='sfs':
                param_grid['feature_selector__neural__sfs__tol'] = [0.001,0.01,0.05,0.1,0.2]


        grid_search = GridSearchCV(pipeline, param_grid, cv=5,scoring='neg_log_loss')
        grid_search.fit(X, y)

        return grid_search.best_estimator_

def get_weights(model,return_dropped_preds=True): 
    """
    Utility function to get the weights from a fitted model
    Used specificalyl for the model constructed above 
    """

    feature_names = model.named_steps['feature_selector'].get_feature_names_out()  
    weights = model.named_steps['logistic_regression'].coef_ 
    intercept =  model.named_steps['logistic_regression'].intercept_

    assert ('bias' in feature_names) & (intercept[0]==0), 'there is a bias parameter, yet the intercept is not 0 ...'

    parameters = pd.DataFrame(weights,columns=feature_names)

    if return_dropped_preds:
        orig_features = model.named_steps['feature_selector'].feature_names_in_
        dropped_features = np.setdiff1d(orig_features,feature_names)
        parameters[dropped_features] = 0
        assert orig_features.size==parameters.size 
        # re-order the features
        parameters = parameters[orig_features]

    # maybe I will change later not to store all the hyperparameters
    all_parameters={
        'weights':parameters, 
        'intercept':intercept[0],
        'hyperparameters': model.get_params()
    }

    
    return all_parameters

def score_model(scorer_name, y_true, y_pred, **kwargs):
    """
    Score a model based on the provided scorer name.

    Parameters:
    - scorer_name (str): The name of the scorer function.
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels or probabilities.
    - **kwargs: Additional keyword arguments passed to the scorer function.

    Returns:
    - float: The score computed by the scorer function.
    """
    scorers = {
        'log_loss': log_loss,
        'accuracy': accuracy_score,
        'f1_score': f1_score,
        'balanced_accuracy_score': balanced_accuracy_score, 
        'roc_auc_score': roc_auc_score
    }

    if scorer_name not in scorers:
        raise ValueError(f"Scorer '{scorer_name}' not found. Available scorers are: {list(scorers.keys())}")

    # Get the scorer function
    scorer_func = scorers[scorer_name]

    # check whether the data contains enough trials
    # if not we will return nan
    if np.unique(y_true).size==1:
        score = np.nan
    else:
        score = scorer_func(y_true, y_pred, **kwargs)


    return score 
# 

def fit_stim_vs_neur_models(trials,neuron_selector_type='sfs'):
    '''
    Fits models for stimulus, neural data, and both combined, returning a 
    dictionary of models, parameters, and training/testing datasets.

    Parameters: 
    trials: DataFrame
        Data containing predictors and target (choice).  

    neuron_selector_type: str, optional
        Type of neuron selector to use in fitting models (default is 'sfs').

    Returns:
        dict

    '''
    # add a bias predictor
    if 'bias' not in trials.columns:
        trials['bias'] = 1

    stim_predictors = ['visR','visL','audR','audL']
    bias_predictors = ['bias']
    neural_predictors = [c for c,_ in trials.items() if 'neuron' in c]

    stim_plus_bias = stim_predictors + bias_predictors
    neur_plus_bias = neural_predictors + bias_predictors
    all_predictors =  stim_predictors + neural_predictors + bias_predictors

    X = trials[all_predictors]
    y = trials['choice']
    stratifyIDs = trials['trialtype_id'] 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1,shuffle=True,stratify=stratifyIDs) # maybe this is the one to save actually

    # now we fit a bunch of models with subsetting the features
    subset_names = ['bias','stim','neur','all']
    feature_names = [bias_predictors,stim_plus_bias,neur_plus_bias,all_predictors]

    X_train_subsets = {name:X_train[feature_set] for name,feature_set in zip(subset_names,feature_names)}
    X_test_subsets = {name:X_test[feature_set] for name,feature_set in zip(subset_names,feature_names)}


    # in the fitting procedure we use different, parameters, hence all hardcoded. 
    models,params = {}, {}
    # bias only 
    models['bias'] = fit_model(X_train_subsets['bias'], y_train, gridCV_vis=False,gridCV_neur=False)
    params['bias'] = get_weights(models['bias'],return_dropped_preds=True)

    # stim only
    models['stim'] = fit_model(X_train_subsets['stim'], y_train, gridCV_vis=True, gridCV_neur=False)
    params['stim'] = get_weights(models['stim'],return_dropped_preds=True)

    # neur-only 
    models['neur'] = fit_model(X_train_subsets['neur'],y_train,gridCV_vis=False, gridCV_neur=True, neuron_selector=neuron_selector_type)
    params['neur'] = get_weights(models['neur'], return_dropped_preds=True)

    # all model, with the gamma power coming from the stim-only model
    stim_gamma  = params['stim']['hyperparameters']['feature_selector__vis__power']
    models['all'] = fit_model(X_train_subsets['all'],y_train, power = stim_gamma, gridCV_vis=False, gridCV_neur=True, neuron_selector=neuron_selector_type)
    params['all'] = get_weights(models['all'],return_dropped_preds=True)


    return {
        'raw': trials,
        'models': models,
        'params': params,
        'X_train': X_train,  # Full X_train saved once
        'X_test': X_test, 
        'X_train_subsets': X_train_subsets,  # Save the subsets once for easy access
        'X_test_subsets': X_test_subsets,    # Save test subsets as well
        'y_train': y_train,
        'y_test': y_test
    }



def get_all_scores(results,
                 scorers = [ 'log_loss','roc_auc_score'],
                 per_trial_type=True
                 ):

    models = results['models']
    X_trains, X_tests = results['X_train_subsets'], results['X_test_subsets']
    y_train, y_test = results['y_train'], results['y_test']

    # check dimensions of each input
    assert len(models)==len(X_trains)==len(X_tests), (
        'model no. does not match training set number'
        )
    assert X_trains['stim'].shape[0]==y_train.size, (
        'the training set size does not match the predictable vector length'
        )

    predictions_train = {
        k:models[k].predict(X_trains[k]) for k in models.keys()
        }
    predictions_test = {
        k:models[k].predict(X_tests[k]) for k in models.keys()
        }
    
    # we compute on training set to assess overfitting 
    scores_train  = {
        f'{scorer}_train':{
            k:score_model(scorer,y_train,predictions_train[k]) 
            for k in models.keys()
            } 
            for scorer in scorers
        }

    # total score on the test set    
    scores_test  = {f'{scorer}_test':{
        k:score_model(scorer,y_test,predictions_test[k]) 
        for k in models.keys()
        } 
        for scorer in scorers
    }
    
    if per_trial_type: 
        visDiff = results['X_test'].visR - results['X_test'].visL
        audDiff = results['X_test'].audR - results['X_test'].audL

        trial_types = get_trial_types(visDiff, audDiff)
        unique_trial_types = np.unique(trial_types)

        # get the scores on specific trial types
        scores_trial_type =  {
            (f'{trial_type}_{scorer}'): {
                k:score_model(
                    scorer,
                    y_test[trial_types==trial_type],
                    predictions_test[k][trial_types==trial_type]
                    ) 
                    for k in models.keys()
                }
                for trial_type,scorer in itertools.product(unique_trial_types,scorers)
        }

        return pd.concat(
            (pd.DataFrame(scores_train), 
             pd.DataFrame(scores_test), 
             pd.DataFrame(scores_trial_type)),
            axis=1
        )


    return pd.concat((pd.DataFrame(scores_train),pd.DataFrame(scores_test)),axis=1)


def session_plots(plot_odds=False,
                    plot_odds_neural = False,
                    plot_AUCs=False,
                    plot_neur_weights=False,
                    plot_stim_weights=False):
    pass

    # if plot_odds:
    #     axs = Odds_hists(models,X_tests,cmap='jet')
    #     # add the LogLik for each model? 
    #     axs[1].set_title('LLstim:%(stim).3f, LLall: %(all).3f' % test_scores)
    

    # if plot_AUCs:
    #     fig = AUCs_compare(models,X_tests,y_test)
    #     #AUCs_compare(models,X_trains,y_train)
    
    # if plot_neur_weights:
    #     RLdiff = (X_train[y_train==1].mean(axis=0)
    #                 -X_train[y_train==0].mean(axis=0))
    #     fig,ax = plt.subplots(1,1,figsize=(5,5))
    #     ax.plot(RLdiff[neural_predictors],params['all']['weights'][neural_predictors].T,'.')
    #     ax.axhline(0, color='black',linewidth=0.5)
    #     ax.axvline(0, color='black',linewidth=0.5)
    #     ax.set_xlabel('R-L firing rate')
    #     ax.set_ylabel('weights')
    #     ax.set_title('neural predictor weights')

    # if plot_stim_weights:
        
        
    #     sensory_terms = pd.concat((params['stim']['weights'],
    #         params['all']['weights'][stim_predictors]))

    #     sensory_terms_ = sensory_terms.reset_index().melt(id_vars='index')
    #     sensory_terms_ = sensory_terms_.rename(columns={'index':'model','value':'weights','variable':'parameters'})

    #     fig,ax = plt.subplots(1,1,figsize=(5,5))

    #     sns.barplot(sensory_terms_,
    #                 x='parameters',
    #                 y='weights',
    #                 hue = 'model',ax=ax)

 

    # if plot_odds_neural:
    #     kappas,betas = plot_neural_psychometric(models['all'],X_tests['all'],y_test)
    #     all_out['kappas'] = kappas
    #     all_out['betas'] = betas
    
def make_fake_data(trials,
                   n_fake_neurons = 10,
                   p_choice_neurons = 0,
                   p_vis_neurons = 0,
                   p_aud_neurons = 0,
                   p_av_neurons = 0,
                   p_v_choice_neurons = 0,
                   p_a_choice_neurons = 0,
                   p_av_choice_neurons = 0,
                   noise_sd = .2):
    
    """ replaces neurons in the trials df. with fake neurons, 
    for the moment only choice neurons
    potentially also add sensory neurons with various gain etc. 

    Parameters:
        trials: pd.df
        n_fake_neurons: float
            how many neurons to create
        p_choice_neurons: float betweeen 0 and 1
            fraction of choice neurons to crete
        noise_sd: float
            how much Gaussian noise to add

    Returns:
        pd.df: the new trials where neurons are replaced with fake ones
    """    
    n_trials = trials.shape[0]

    # possible neuron options
    n_choice = trials.choice.values
    n_vis = trials.visDiff.values
    n_aud = trials.audDiff.values
    n_av  = n_vis * n_aud
    n_v_choice = n_choice * n_vis
    n_a_choice = n_choice * n_aud
    n_av_choice = n_choice * n_aud * n_vis

    n_random = np.random.choice([1,0],size=n_trials,p=[0.5,0.5])

    # 
    p_random_neurons = 1 - np.sum((
        p_choice_neurons,
        p_vis_neurons,
        p_aud_neurons,
        p_av_neurons,
        p_v_choice_neurons,
        p_a_choice_neurons,
        p_av_choice_neurons 
    ))

    # check whether the ratios are feasible
    assert np.sum((p_choice_neurons,
        p_vis_neurons,
        p_aud_neurons,
        p_av_neurons,
        p_v_choice_neurons,
        p_a_choice_neurons,
        p_av_choice_neurons,
        p_random_neurons
        )) ==1,'Probabilities do not sum to 1.'

    # generate the neuron combinations
    neuron_types = np.random.choice(
        ['c','v','a','av','cv','ca','cav','r'],
        size=n_fake_neurons,
        p=[p_choice_neurons,
           p_vis_neurons,
           p_aud_neurons,
           p_av_neurons,
           p_v_choice_neurons,
           p_a_choice_neurons,
           p_av_choice_neurons,
           p_random_neurons]
        )

    neurons={}
    for idx,n in enumerate(neuron_types):
        if n=='c': c_y = n_choice 
        elif n=='v': c_y = n_vis
        elif n=='a': c_y = n_aud
        elif n=='av': c_y = n_av
        elif n=='cv': c_y = n_v_choice
        elif n=='ca': c_y = n_a_choice
        elif n=='cav': c_y = n_av_choice        
        elif n=='r': c_y = n_random

        # I think I don't need this if I zscore
        random_baseline_fr = np.random.uniform(1.0, 50.0) 
        gaussian_noise = np.random.normal(loc = 0, 
                                          scale = noise_sd, 
                                          size = c_y.size)
        
        neurons[('neuron_%.0f' % idx)] =  zscore(c_y + 
                                                 random_baseline_fr + 
                                                 gaussian_noise)

    neurons = pd.DataFrame(neurons)

    # construct the new df
    non_neuron_columns = trials.drop(columns=trials.filter(like='neuron_').columns)

    fake_neurons_trials = pd.concat(
        (non_neuron_columns,neurons),axis=1
    )

    return fake_neurons_trials

def fit_parse_(rec,nametag=None,**fit_kwargs):
    """helper fuction to fit the neurometric model and collect the key metrics of the results

    Args:
        trials (pathlib.Path): path to csv

    Returns:
        pd.df: results of fitting
    """

    if isinstance(rec, (str, Path)):
         print('fitting', rec, '...')
         trials = pd.read_csv(rec)  # Load trials from the CSV file
         subject = rec.name.split('_')[0]

    elif isinstance(rec, pd.DataFrame):
        trials = rec  # Use the DataFrame directly 
        if nametag: subject=nametag
        else: subject = 'test'
        
    else:
        raise ValueError("input must be either str,Path, or pd.df") 
    
    results = fit_stim_vs_neur_models(trials,**fit_kwargs)

    
    scores = get_all_scores(results)
    scores = scores.reset_index()
    scores.rename(columns = {'index': 'model'}, inplace=True)
    pivoted_data = scores.set_index('model').unstack().to_frame().T
    pivoted_data.columns = ['_'.join(col).strip() for col in pivoted_data.columns]


    session_info = pd.DataFrame({
    'subject': [subject],
    'nTrials': results['X_test'].shape[0],
    'nNeur': results['X_test'].shape[1] - 4,
    'nNeur_used': np.sum(results['params']['all']['weights'].values != 0) - 4,
    'pCorrect': np.mean(trials.feedback == 1),
    '|initBias|':  results['params']['stim']['weights']['bias'].values[0]
    })

    # Concatenate with session info to create a single row per dataset
    final_data = pd.concat([session_info, pivoted_data], axis=1)

    return final_data
    

def batch_fit_neurometric(all_files, neuron_selector_type, savepath=None):
    """
    Process all records and compile neurometric data into a DataFrame.
    """
    fit_kwargs = {
        'neuron_selector_type': neuron_selector_type
    }

    processed_data = [fit_parse_(rec,**fit_kwargs) for rec in all_files]
    
    # Create a DataFrame from the processed data
    df = pd.concat(processed_data)

    # all the recordings and mice etc.

    if savepath:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        fit_result = savepath.parent / ('result_%s_%s.csv' % (neuron_selector_type,timestamp))
        df.to_csv(fit_result)

    return df