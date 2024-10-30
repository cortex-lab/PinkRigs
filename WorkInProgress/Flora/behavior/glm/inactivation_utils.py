


from pathlib import Path 
import pandas as pd 
import numpy as np
from scipy import stats


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import log_loss,roc_auc_score 
from predChoice_utils import fit_model,get_weights


class LogisticRegression_partialPenalty(SGDClassifier):
    """
    Logistic Regression with Stochastic Gradient Descent
    Customised function such that we are able to apply penalty separately on the opto terms

    """
    def __init__(self,is_penalised_feature=None,alpha=0,**kwargs):
        super().__init__(penalty=None,loss = 'log_loss',**kwargs)
        self.is_penalised_feature = is_penalised_feature
        self.alpha = alpha

    def _apply_custom_penalty(self,coef):
        if self.is_penalised_feature is None: 
            return coef
        penalty = np.zeros_like(coef)
        penalty[:,self.is_penalised_feature] = self.alpha * np.sign(coef[:,self.is_penalised_feature])
        #penalty[:,self.is_penalised_feature] = self.alpha * coef[:,self.is_penalised_feature]
        return coef - penalty

    def partial_fit(self,X,y,classes=None):
        super().partial_fit(X,y,classes)
        self.coef_ = self._apply_custom_penalty(self.coef_)

        return self

    def fit(self,X,y):
        super().fit(X,y)
        self.coef_ = self._apply_custom_penalty(self.coef_)
        return self

#  can build the piping specifically for fitting the gamma here

def fit_opto_model(rec,nametag=None,gammafit=False,L2opto=None):
    
    if isinstance(rec, (str, Path)):
        print('fitting', rec, '...')
        trials = pd.read_csv(rec)  # Load trials from the CSV file
        subject = rec.name.split('_')[0]

    elif isinstance(rec, pd.DataFrame):
        trials = rec  # Use the DataFrame directly 
        if nametag: subject=nametag
        else: subject = 'test'


    stim_predictors = ['visR','visL','audR','audL','bias']
    opto_predictors = ['visR_opto','visL_opto','audR_opto','audL_opto','bias_opto']    

    all_predictors =  stim_predictors + opto_predictors

    #filter the trial matrix
    trials = trials[(trials.choice==0) | (trials.choice==1)] # keep only the post-stim correct trials

    X = trials[all_predictors]
    y = trials['choice']
    stratifyIDs = trials['trialtype_id']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1,shuffle=True,stratify=stratifyIDs)

    if gammafit:
        m = fit_model(X,y,gridCV_vis=True)
        w = get_weights(m)
        params = w['weights']
        params['gamma'] = None
    else:
        
        if L2opto is not None:
            m = LogisticRegression_partialPenalty(
                is_penalised_feature = np.isin(X.columns,opto_predictors),alpha = L2opto, fit_intercept = False)
        else: 
            m = LogisticRegression(fit_intercept=False)

        m.fit(X_train,y_train)
        params = pd.DataFrame(m.coef_,columns=all_predictors)

    y_pred_prob = m.predict_proba(X_test)
    y_pred = m.predict(X_test)
    neg_log_loss = -log_loss(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test,y_pred)

    return params,neg_log_loss,auc_score