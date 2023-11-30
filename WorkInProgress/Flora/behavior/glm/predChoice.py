# %% 

# prototype glmfit


# todo: figure out closs validation
# figure out: param contribution evaluation



import sys
import numpy as np
import pandas as pd  
from scipy.optimize import minimize

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import format_events

class AVSplit(): 
    """
    example model object that takes certain conditions & parameters and spits out the log odds

    for each class we need to mass the names of the parameters     
    """
    required_parameters = ["aL", "aR","vL","vR","gamma","bias"]
    required_conditions = ["audDiff","visDiff"]


    def get_logOdds(self,conditions,**args):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        
        a_R = (audSide>0)
        a_L = (audSide<0)
        v_R = (visSide>0) * (visContrast**self.gamma)
        v_L = (visSide<0) * (visContrast**self.gamma)
        audComponent = (self.aR) * a_R - (self.aL) * a_L
        visComponent = (self.vR) * v_R - (self.vL) * v_L
        biasComponent = self.b 

        return  audComponent + visComponent + biasComponent

def format_av_trials(ev,spikes=None):
    """
    specific function for the av pipeline such that the _av_trials.table is formatted for the glmFit class


    Parameters: 
    ----------
    ev: Bunch
        _av_trials.table
    spikes: Bunch 
        default output of the pipeline
      
    todo: input format contains spikes

    Returns: pd.DataFrame
    """

    ev = format_events(ev)
    maxV = np.max(np.abs(ev.visDiff))
    maxA = np.max(np.abs(ev.stim_audAzimuth))

    df = pd.DataFrame()
    df['visDiff']=ev.visDiff/maxV
    df['audDiff']=ev.stim_audAzimuth/maxA
    df['choice'] = ev.response_direction-1

    # filtering of the data
    # by default we get rid of 
        # nogos  
        # invalid trials (i.e. repeatNum!=1)
        # 30 degree aud azimuth
    # we also can optionally get rid of other trials later... 
    keep_trial_idx = ((ev.is_validTrial) & 
                      (ev.response_direction!=0) & 
                      (np.abs(ev.stim_audAzimuth)!=30))
    
    df = df[keep_trial_idx].reset_index()

    return df


class glmFit(): 
    def __init__(self,trials,cv_type=None):
        """
        function to that checks whether fit can be correctly initialised given the input data.
        Parameters:
        ----------
        trials: pd.DataFrame
            table where each row is trial, columns can be:
                choice (required at all times) i.e. the y (predicted value)
                etc. that will all be treated as predictors (required, given the model, e.g. audDiff,visDiff)
        cv_type: 
            type of cv splitting, default StratifiedCVsplit
        """

        assert 'choice' in trials.columns, 'choice is missing.'

        "X: predictors, y = choices"

        predictors = trials.drop('choice',axis='columns')

        self.predictor_names = predictors.columns
        self.X = predictors.values
        self.y = trials.choice 


    def LogOddsModel(self,model_type='AVSplit',params=None):
        """
        function to calculate logOdds for each trial, given the model and its parameters
        Parameters:
        ----------
        model_type: str 
        params: dict? 

        Returns: 
        ---------
            np.ndarray
            log odds

        todo: redefine model when model contribution is assessed (i.e. fixedparam/freeP business)
        """
        if model_type=='AVSplit':
            model = AVSplit()

        params = {"aL":1, "aR":1,"vL":1,"vR":1,"gamma":1,"bias":1}    
        model.get_logOdds(self.X,params)


    def caluclatepHat(self,**modelparams):
        """
        calculate the probability of making each possible choice given the model 
        """
        logOdds = self.LogOddsModel(**modelparams)
        pR = np.exp(logOdds) / (1 + np.exp(logOdds))
        pHat = np.array([pR,1-pR])        
        return pHat
    
    def get_Likelihood(self,testParams): 
        pHat_calculated = self.calculatepHat(testParams) # the probability of each possible response 
        responseCalc = self.dataBlock['response_direction'] # the actual response taken
        
        # calculate how likely each of these choisen response was given the model  (atm bullshit)  
        # return its negativelog2likelihood      
        logLik = -np.mean(np.log2(pHat_calculated[np.arange(pHat_calculated.shape[0]), responseCalc]))
        return logLik 
    
    def fit(self):
        """
        fit the model by minimising the logLikelihood
        i.e. the get_Likelihood function  

        todo: optimse parameters for search
        """

        fittingObjective = lambda b: self.get_Likelihood(b)
        result = minimize(fittingObjective, self.prmInit, bounds=self.prmBounds) # one could also specify solver etc. 


    def predict():
        pass 

    def fit_predict(): 
        pass 

    def visualise(): 
        pass 



# %%
