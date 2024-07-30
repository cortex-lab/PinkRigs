#%%
# this code will be the same as PredChoice, but the fitting procedure will be fully using scikit-learn
# key things I will implement: grid-search for gamma
# train validation and test sets
# maybe apply it to all the data??

# 
# preprocess the PinkRigs data - output is a df 

# generic libs 
from numbers import Integral, Real
import numpy as np
import pandas as pd
from pathlib import Path 

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn libs
from sklearn.base import clone,is_classifier
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedShuffleSplit,check_cv
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from predChoice_utils import gamma_transform

# data read in
savepath = Path(r'D:\LogRegression')
all_files = list(savepath.glob('*.csv'))

rec = all_files[0]
trials = pd.read_csv(rec)


pred_list = [c for c,_ in trials.items() if 'neuron' in c]
pred_list = ['visR','visL','audR','audL'] + pred_list

# pred_list = ['visR','visL','audR','audL','neuron_142','neuron_1350','neuron_1351','neuron_1335']

X = trials[pred_list]
y = trials['choice']
stratifyIDs = trials['trialtype_id']

# model fitting pipeline 

class CustomSequentialFeatureSelector(SequentialFeatureSelector):
    '''
    Custom Sequential feature selector that can keep some features deterministically by default, 
    while performing sequential feature selection on the other ones. 
    Essentially a modification of the sklearn.feature_selection.SequentialFeatureSelector, 
    where I copy and modify the original fit function, such that the starting current mask is not zeros 
    but instead is masked with always_keep mask
    
    Parameters: 
    -----------

    Attributes:
    -----------

    Returns:
    --------

    '''
    def __init__(self, estimator, 
                 always_keep=None,
                 n_features_to_select="auto",
                tol=None,
                direction="forward",
                scoring=None,
                cv=5,
                n_jobs=None):

        super().__init__(estimator)
        self.n_features_to_select = n_features_to_select
        self.tol = tol
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.always_keep = always_keep
    
    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        tags = self._get_tags()

        n_features = X.shape[1]

        # we initiate a mask - based on what is selected to be always kept
        if self.always_keep is not None: 
            current_mask = np.isin(X.columns,self.always_keep)
            assert sum(current_mask)==len(self.always_keep), 'some columns that are expected to be kept are missing' 
        else:
            current_mask = np.zeros(shape=n_features, dtype=bool)
            

        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        
        if self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features - 1
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if self.n_features_to_select >= n_features:
                raise ValueError("n_features_to_select must be < n_features.")
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection

        n_iterations = (
            self.n_features_to_select_
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"
        for _ in range(n_iterations):
            new_feature_idx, new_score = self._get_best_new_feature_score(
                cloned_estimator, X, y, cv, current_mask
            )
            if is_auto_select and ((new_score - old_score) < self.tol):
                break

            old_score = new_score
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self
    
def run_fit_pipeline(X,y):

    pipeline = Pipeline([
        ('gamma_transformer',FunctionTransformer(func=gamma_transform)),
        # ('feature_selector',CustomSequentialFeatureSelector(LogisticRegression(),
        #                                                     always_keep=['visR','visL','audR','audL']
        #                                                     )),
        ('logistic_regression',LogisticRegression())
    ])

    #gamma_options = np.arange(.2,3,.1)
    gamma_options = np.array([1,1.5])

    param_grid = {
        'gamma_transformer__kw_args':[{'gamma':g} for g in gamma_options],
        #'feature_selector__tol':[5,10],
       # 'feature_selector__scoring':['neg_log_loss'] # this is -(-Log2Likelihood) i.e. it needs to be maximised

        # then maybe we can test a tol arg
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_log_loss',
        cv=StratifiedShuffleSplit(n_splits=2,test_size=.2,random_state=1)) 

    model.fit(X,y)

    return model
 
# %%
# visualisation plots 

# some data formatting utility 

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

def get_params(model,X):
    gamma = np.array([model.best_params_['gamma_transformer__kw_args']['gamma']])
    best_model = model.best_estimator_
    selected_features = best_model.named_steps['feature_selector'].get_support()
    coefficients = best_model.named_steps['logistic_regression'].coef_
    intercept = best_model.named_steps['logistic_regression'].intercept_    
    params = pd.DataFrame(coefficients,columns=X.iloc[:,selected_features].columns)
    params['gamma'] = gamma
    params['bias'] = intercept
    return params

def get_LogOdds(model,X,columns=None):
    '''
    custom calculating of logOdds, such that not all the predictors are used necessarily
    Parameters: 
    ---------
    model: sklearn- class
    X: pd.df
    columns: np.ndarray
        determines which columns to keep  when calculating the odds

    '''
    if columns is None: 
        columns = np.ones(X.shape[1]).astype('bool')
       # logOdds = model.decision_function(X) 

    params = get_params(model,X)
    X_ = X.iloc[:,columns]
    X_ = gamma_transform(X_,gamma=params.gamma.values)
    feature_selector = model.best_estimator_.named_steps['feature_selector']
    X_ = X_.iloc[:,feature_selector.get_support()]
    kept_predictors  = X_.columns
    kept_weights = params.iloc[:,np.isin(params.columns.values,kept_predictors.values)]
    assert X_.columns.equals(kept_weights.columns) # check if columns are in correct order
    logOdds = np.dot(X_,kept_weights.T) + params.bias.values
    logOdds = np.ravel(logOdds)

    return logOdds 

def raise_to_sigm(logOdds):
    'function to calculate pR from logOdds'
    return np.exp(logOdds) / (1 + np.exp(logOdds))

# plotting functions
def plot_Odds(logOdds,
              minOdd=None,maxOdd=None,
              ax=None,axh=None,color='#E1BE6A'):
    
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

    # the sigmoid curve
    odds = np.linspace(minOdd,maxOdd,100)
    ax.plot(odds,raise_to_sigm(odds),color='k',alpha=.5)
    # th actual models predictions


    bin_pos = np.linspace(minOdd,maxOdd,8)
    mid_bins,pR_per_bin = get_pR_per_bin(logOdds,bin_pos)
    ax.scatter(mid_bins,pR_per_bin,
            s=30,edgecolors='k',c=color)

    axh.hist(logOdds,bins=bin_pos,
            rwidth=.9,alpha=.7,align='mid',color=color)
    
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

    axh.axvline(0,color='k',linestyle='--')
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0.5,color='k',linestyle='--')

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

def Odds_hists(models,Xs,cmap='viridis'):

    _,(ax,axh) = plt.subplots(2,1,figsize=(6,10),
                                sharex=True,
                                gridspec_kw={'height_ratios':[3,1]})
    
    if len(models)==1:
        stim_Odds = get_LogOdds(models[0],Xs[0],columns=get_stim_predictors(X))
        neural_Odds = get_LogOdds(models[0],Xs[0],columns=None)
        LogOdds = [stim_Odds,neural_Odds]
    else:
        LogOdds = [get_LogOdds(model,X,columns=None) for model,X in zip(models,Xs)]

    odd_plot_kws ={
        'minOdd':min(np.concatenate((LogOdds))),
        'maxOdd':max(np.concatenate((LogOdds))),
        'ax':ax,
        'axh':axh
    }
    colormap = plt.cm.get_cmap(cmap)

    # Use the colormap
    colors = colormap(np.linspace(.2, .8, 2))

    [plot_Odds(l_o,color=my_color,**odd_plot_kws) for l_o,my_color in zip(LogOdds,colors)]

def AUCs_compare(X1,y,model1,model2=None,X2=None):
    
    if model2 is None:
        model1_Odds = get_LogOdds(model1,X1,columns=get_stim_predictors(X))
        model2_Odds = get_LogOdds(model1,X1,columns=None)
    else: 
        model1_Odds = get_LogOdds(model1,X1,columns=None)
        model2_Odds = get_LogOdds(model2,X2,columns=None)
    
    visDiff,audDiff = get_stimDiffs(X1)

    plot_AUCs(
        model1_Odds,model2_Odds,
        visDiff,audDiff,y
        )

def get_scores_per_trialType(model,X,y):
    visDiff,audDiff = get_stimDiffs(X)
    trialTypes = get_trial_types(visDiff,audDiff)
    unique_trials = np.unique(trialTypes)
    scores = {t:[model.score(X[trialTypes==t],y[trialTypes==t])] for t in unique_trials}
    return pd.DataFrame(scores)

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1,shuffle=True,stratify=stratifyIDs)

model = run_fit_pipeline(X_train,y_train)

# %%  pipelinenize


kept_pred_sets = {
    'stim':get_stim_predictors(X_train), # stim only 
    'neur':np.ones(X_train.shape[1]).astype('bool')
}

X_trains,X_tests,models,train_scores,test_scores = {},{},{},{},{}
for pred_set in kept_pred_sets.keys():
    is_sel = kept_pred_sets[pred_set]
    X_train_ = X_train.iloc[:,is_sel]
    X_test_ = X_test.iloc[:,is_sel]
    model = run_fit_pipeline(X_train_,y_train)
    plot_psychometric(X_train_,y_train,model,yscale='logS')
    models[pred_set] = model
    X_trains[pred_set] = X_train_
    X_tests[pred_set] = X_test_
    train_scores[pred_set] = get_scores_per_trialType(model,X_train_,y_train)
    test_scores[pred_set] = get_scores_per_trialType(model,X_test_,y_test)


Odds_hists(models.values(),X_tests.values(),cmap='jet')
AUCs_compare(X_tests['stim'],y_test,model1=models['stim'],model2=models['neur'],X2=X_tests['neur'])

# %%
