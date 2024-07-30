
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel,VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# first fit the neural only model 

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


def fit_model(X,y,power=1,gridCV_vis=False,gridCV_neur=False):

    neural_transformer = Pipeline([
        ('variance_thr',VarianceThreshold(threshold=0.001)),
        ('lasso',SelectFromModel(LogisticRegression(
            penalty='l1',solver='liblinear'),threshold=0.2))
        ])

    neural_predictors = [c for c,_ in X.items() if 'neuron' in c]
    is_neural_predictor = np.isin(X.columns,neural_predictors)
    is_vis_predictor = np.isin(X.columns,['visL','visR'])
    
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
        ('logistic_regression',LogisticRegression())
    ])

    if (not gridCV_vis) and (not gridCV_neur):

        pipeline.fit(X,y)    
        return pipeline
    
    else:
        
        param_grid = {}
        if gridCV_vis:
            param_grid['feature_selector__vis__power'] = np.round(np.arange(0.1,2,0.1),2)
        
        if gridCV_neur:
            param_grid['feature_selector__neural__lasso__threshold'] = [0.01,0.05,0.1,0.2,0.3,0.5,1]


        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
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
        'bias':intercept[0],
        'hyperparameters': model.get_params()
    }

    
    return all_parameters
