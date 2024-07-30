# %% 

# prototype glmfit
# todo: figure out closs validation
# figure out: param contribution evaluation

import sys,re
import numpy as np
import pandas as pd  
from scipy.optimize import minimize
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import format_events,simplify_recdat
from Analysis.neural.utils.spike_dat import get_binned_rasters
from Analysis.neural.utils.data_manager import load_cluster_info
from Analysis.pyutils.ev_dat import filter_active_trials

class AVSplit(): 
    """
    example model object that takes certain conditions & parameters and spits out the log odds
    for each class we need to mass the names of the parameters  
    subfunctions:   
    """

    required_parameters = np.array(["aL", "aR","vL","vR","gamma","bias"])

    required_conditions = np.array(["audDiff","visDiff"])

    def __init__(self,is_neural=False,fixed_parameters = [1,1,1,1,1,1],fixed_paramValues=[1,1,1,1,1,0]):
        """
        optional initialisation for neural model
        """
        self.is_neural = is_neural
        self.fixed_parameters  = np.array(fixed_parameters)
        self.fixed_paramsValues = np.array(fixed_paramValues)
    
    def get_all_params(self,betas):
        """
        construct parameter arrays (e.g. that feeds into get_logOdds) from the betas (fittables) and the fixed parameters
        """

        n_fixed = np.sum(self.fixed_parameters.astype('bool'))
        n_free = np.sum(~self.fixed_parameters.astype('bool'))
        if n_fixed:

            free_values = np.zeros(self.fixed_parameters.size)
            free_values[~self.fixed_parameters.astype('bool')] = betas[:n_free]

            fixed_values = self.fixed_parameters * self.fixed_paramsValues

            all_params  = (free_values + fixed_values) 
            neural = betas[n_free:]
            returned_params = np.concatenate((all_params, neural))

        else: 
            returned_params = betas
        
        return returned_params 

    def get_logOdds(self,conditions,parameters):
        
        nTrials = conditions.shape[0]
        if self.is_neural:
            neural_parameters = parameters[self.required_parameters.size:] # each neuron has its own parameter
            neural_conditions = conditions[:,self.required_conditions.size:]
            non_neural_parameters = parameters[:self.required_parameters.size]
            non_neural_conditions = conditions[:,:self.required_conditions.size]
            neural_contribution = np.matmul(neural_conditions,neural_parameters)

        else: 
            non_neural_parameters,non_neural_conditions = parameters,conditions
            neural_contribution = 0

        aL = non_neural_parameters[self.required_parameters=='aL']
        aR = non_neural_parameters[self.required_parameters=='aR']
        vL = non_neural_parameters[self.required_parameters=='vL']
        vR = non_neural_parameters[self.required_parameters=='vR']
        gamma = non_neural_parameters[self.required_parameters=='gamma']
        bias = non_neural_parameters[self.required_parameters=='bias']

        visDiff = non_neural_conditions[:,self.required_conditions=="visDiff"]
        audDiff = non_neural_conditions[:,self.required_conditions=="audDiff"]
        visContrast = np.abs(visDiff)
        visSide = np.sign(visDiff)
        audSide = np.sign(audDiff)
        
        a_R = (audSide>0)
        a_L = (audSide<0)
        v_R = (visSide>0) * (visContrast**gamma)
        v_L = (visSide<0) * (visContrast**gamma)
        audComponent = (aR) * a_R - (aL) * a_L
        visComponent = (vR) * v_R - (vL) * v_L
        biasComponent = bias 

        return  np.ravel(audComponent + visComponent + biasComponent) + neural_contribution
    
    @staticmethod
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


    @staticmethod
    def get_trial_types(visDiff,audDiff):
        """
        function that retunrns an np.array of strings based that will indicate what type of trial we are in (vis/aud/coh/conflict/blank)
        """
        is_blank,is_vis,is_aud,is_coh,is_conf= AVSplit.get_trial_bools(visDiff,audDiff)


        trial_types = np.empty(visDiff.shape, dtype='U8')
        trial_types[is_blank]=['blank']
        trial_types[is_vis]=['vis']
        trial_types[is_aud]=['aud']
        trial_types[is_coh]=['coherent']
        trial_types[is_conf]=['conflict']

        return trial_types


            
    @staticmethod
    def get_ll_per_condition(ll):
        """
        function to return the log likelihoods per condition given the matrix that is outputted by each unique split 
        """

        is_blank,is_vis,is_aud,is_coh,is_conf = AVSplit.get_trial_bools(ll.visDiff,ll.audDiff)


        ll_blank = ll[is_blank].LogLik.mean()
        ll_aud =  ll[is_aud].LogLik.mean()       
        ll_vis =  ll[is_vis].LogLik.mean()       
        ll_coh =  ll[is_coh].LogLik.mean()       
        ll_conf =  ll[is_conf].LogLik.mean() 

        return ll_blank,ll_aud,ll_vis,ll_coh,ll_conf
    
    
    def get_predictions(self):
        pass
        

    def plot(self,parameters,yscale='log',conditions=None,choices=None,ax=None,colors=['b','grey','red'],dataplotkwargs={'marker':'o','ls':''},predpointkwargs ={'marker':'*','ls':''},predplotkwargs={'ls':'-'}):
        """
        plot the model prediction for this specific model
        if the model has neural components we 0 those out
        """
        if ax is None:
            _,ax = plt.subplots(1,1,figsize=(8,8))
        
        if self.is_neural:  
            non_neural_params = parameters[:self.required_parameters.size]
            non_neural_conds = conditions[:,:self.required_conditions.size]
            n_neurons = conditions.shape[1]-self.required_conditions.size
        else: 
            non_neural_params,non_neural_conds = parameters,conditions


        if (conditions is not None) & (choices is not None):
            visDiff = np.ravel(non_neural_conds[:,self.required_conditions=="visDiff"])
            audDiff = np.ravel(non_neural_conds[:,self.required_conditions=="audDiff"])
            Vs = np.unique(visDiff)
            As = np.unique(audDiff)

            Vmesh,Amesh =np.meshgrid(Vs,As)
            for v,a,mycolor in zip(Vmesh,Amesh,colors):
                x = v
                x = np.sign(x)*np.abs(x)**non_neural_params[self.required_parameters=='gamma']
                y  = np.array([np.mean(choices[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
                
                logOdds =self.get_logOdds(conditions,parameters)
                pR = np.exp(logOdds) / (1 + np.exp(logOdds))

                y_pred = np.array([np.mean(pR[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
                if yscale=='log':
                    y =np.log(y/(1-y))
                    y_pred = np.log(y_pred/(1-y_pred))

                ax.plot(x,y,color=mycolor,**dataplotkwargs)
                ax.plot(x,y_pred,color=mycolor,**predpointkwargs)



        #plotting the prediciton psychometric w\o the neural values  
        nPredPoints = 600  
        Vmodel = np.linspace(-1,1,nPredPoints)
        x_ = np.sign(Vmodel)*np.abs(Vmodel)**non_neural_params[self.required_parameters=='gamma']
        Amodel = np.linspace(-1,1,3)
        for a,mycolor in zip(Amodel,colors):
            conds = np.array((np.ones((nPredPoints))*a,Vmodel)).T
            
            if self.is_neural:
                zeroM = np.zeros((nPredPoints,n_neurons))
                conds = np.concatenate((conds,zeroM),axis=1)

            y_ = self.get_logOdds(conds,parameters)
            if yscale!='log':
                y_ = np.exp(y_) / (1 + np.exp(y_))
            ax.plot(x_,y_,color=mycolor,**predplotkwargs)

        # plot the predicted probabilities alone if there is neural data too...


        if yscale=='log':
            ax.axhline(0,color='k',ls='--')
        else:
            ax.axhline(.5,color='k',ls='--')

        ax.axvline(0,color='k',ls='--')

    def plotPred(self,parameters,yscale='log',conditions=None,choices=None,ax=None,colors=['b','grey','red']):
        """
        plot the model prediction for this specific model
        if the model has neural components we 0 those out
        """
        if ax is None:
            _,ax = plt.subplots(1,1,figsize=(8,8))
        
        if self.is_neural:  
            non_neural_params = parameters[:self.required_parameters.size]
            non_neural_conds = conditions[:,:self.required_conditions.size]
            n_neurons = conditions.shape[1]-self.required_conditions.size
        else: 
            non_neural_params,non_neural_conds = parameters,conditions


        if (conditions is not None) & (choices is not None):
            visDiff = np.ravel(non_neural_conds[:,self.required_conditions=="visDiff"])
            audDiff = np.ravel(non_neural_conds[:,self.required_conditions=="audDiff"])
            Vs = np.unique(visDiff)
            As = np.unique(audDiff)

            Vmesh,Amesh =np.meshgrid(Vs,As)
            for v,a,mycolor in zip(Vmesh,Amesh,colors):
                x = v
                x = np.sign(x)*np.abs(x)**non_neural_params[self.required_parameters=='gamma']
                y  = np.array([np.mean(choices[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
                
                logOdds =self.get_logOdds(conditions,parameters)
                
                pR = np.exp(logOdds) / (1 + np.exp(logOdds))
                
                #ax.hist(logOdds)

                y_pred = np.array([np.mean(pR[(visDiff==vi) & (audDiff==ai)]) for vi,ai in zip(v,a)])
                if yscale=='log':
                    y =np.log(y/(1-y))
                    y_pred = np.log(y_pred/(1-y_pred))

                #ax.plot(y,y_pred,color=mycolor,**predpointkwargs)
                ax.scatter(y,y_pred,s=np.arange(1,y.size+1)**2.5+50,color=mycolor,edgecolor='k')
                #ax.plot(y,y_pred,color=mycolor)
                ax.axline((0,0),slope=1,color='k',linestyle='--')
                ax.set_xlabel('actual')
                ax.set_ylabel('predicted')
                ax.set_title('r = %.2f' % np.corrcoef(y,y_pred)[0,1]) 


        


def format_av_trials(ev,spikes=None,cam=None,nID=None,single_average = False,pre_time=0.2,post_time=0, onset_time = 'timeline_audPeriodOn',**kwargs):
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
    df['feedback'] = ev.response_feedback
    
    if post_time is not None:
        rt_params = {'rt_min':post_time+0.03,'rt_max':1.5}
    else:
        rt_params = {'rt_min':0.03,'rt_max':1.5}


    to_keep_trials = filter_active_trials(ev,rt_params=rt_params,**kwargs)
       
   
    # add choice related activity of it was requested
    if spikes: 
        # tbd
        raster_kwargs = {
                'pre_time':pre_time,
                'post_time':post_time, 
                'bin_size':pre_time+post_time,
                'smoothing':0,
                'return_fr':True,
                'baseline_subtract': False, 
        }

        
        t_on = ev[onset_time]

        # this only works if all t_ons are nans which is ofc not true always
        r = get_binned_rasters(spikes.times,spikes.clusters,nID,t_on[~np.isnan(t_on)],**raster_kwargs)
        
        if single_average: 
            r.rasters = r.rasters.mean(axis=1)[:,np.newaxis,:]
        # zscore across the trials so that the neurons that do not vary per trial do not get added as baseline weights    
        zscored = zscore(r.rasters[:,:,0],axis=0) 

        # discard neurons that are nan on all trials that were kept 
        discard_idx =  np.isnan(zscored).any(axis=0)

        # get back the nans when t_on was nan
        resps = np.empty((t_on.size,zscored.shape[1]))*np.nan
        resps[~np.isnan(t_on),:] = zscored
     
        # some more sophisticated cluster selection as to what goes into the model
        
        if single_average:
            #df['neuron'] = pd.DataFrame((resps[:,~discard_idx].mean(axis=1))) 
            df['neuron']  = pd.DataFrame(resps[:,~discard_idx])
        else:
            nrnNames  = np.array(['neuron_%.0d' % n for n in nID])[~discard_idx]
            df[nrnNames] = pd.DataFrame(resps[:,~discard_idx])


    if cam:
        from Analysis.pyutils.video_dat import get_move_raster
        # get all auditory stimulus onsets
        bin_kwargs  = {
            'pre_time':pre_time,
            'post_time':post_time, 
            'bin_size': pre_time+post_time,
            'sortAmp':False, 'to_plot':False,
            'baseline_subtract':False
        }


        cam_values = (cam.ROIMotionEnergy) # or maybe I should do things based on PCs
        t_on = ev[onset_time]
        move_raster,_,_  = get_move_raster(t_on[~np.isnan(t_on)],cam.times,cam_values,**bin_kwargs) 
        
        zscored = zscore(move_raster,axis=0) 

        resps = np.empty((t_on.size,zscored.shape[1]))*np.nan
        resps[~np.isnan(t_on),:] = zscored
        df['movement'] = pd.DataFrame(resps)
        

        # also do it for each PC
        nPCs = 100 
        PCs_raster,_,_ = zip(*[get_move_raster(t_on[~np.isnan(t_on)],cam.times,cam._av_motionPCs[:,0,i],**bin_kwargs) for i in range(nPCs)])
        PCs_raster = np.concatenate(PCs_raster,axis=1)
        PCs_raster_ = np.empty((t_on.size,nPCs))*np.nan
        PCs_raster_[~np.isnan(t_on),:] = PCs_raster
        
        for i in range(nPCs):
            df['movement_PC%.0d' % i] = PCs_raster_[:,i]



    df = df[to_keep_trials].reset_index(drop=True)

    return df
  


class glmFit(): 

    def __init__(self,trials,model_type='AVSplit',groupby=None,alpha  = 0,**logOddsPars):
        """
        function to that checks whether fit can be correctly initialised given the input data.
        Parameters:
        ----------
        trials: pd.DataFrame
            table where each row is trial, columns can be:
                choice (required at all times) i.e. the y (predicted value)
                etc. that will all be treated as predictors (required, given the model, e.g. audDiff,visDiff)
        groupby: str
            when several types of sessions are fitted together, this parameter indexes into the trials
        cv_type: 
            type of cv splitting, default StratifiedCVsplit
        """

        assert 'choice' in trials.columns, 'choice is missing.'

        "X: predictors, y = choices"

        predictors = trials.drop('choice',axis='columns')
        self.predictor_names = list(predictors.columns)

        self.alpha = alpha
        
        # sepearate the neural predictors
        is_neural_predictor = np.array(['neuron' in p or 'movement' in p for p in self.predictor_names])
        is_neural_model = any(is_neural_predictor)
        if is_neural_model:
            self.neurons = predictors.values[:,is_neural_predictor]
            non_neural_predictors  = list(predictors.columns[~is_neural_predictor])
            self.predictor_names = non_neural_predictors
            self.n_neurons = sum(is_neural_predictor)
        else:
            self.neurons,self.n_neurons = None,0

        self.model = self.LogOddsModel(model_type,is_neural=is_neural_model,**logOddsPars)     
        
        # reorder x such that columns are ordered as required by model
        pred_ = np.concatenate([predictors[p].values[:,np.newaxis] for p in self.model.required_conditions  if p!='neuron'],axis=1)
        self.non_neural_predictors = pred_

        # concatenate with the neural predictors that always go behind the other required predictors for the model
        if is_neural_model:
            pred_ = np.concatenate((pred_,self.neurons),axis=1)

        self.conditions = pred_
        self.choices = trials.choice.values

    def generate_param_matrix():
        # used when fitting s everal session types together (i.e. when certain parameters are fixed across sessions while others are modular
        pass 

    def LogOddsModel(self,model_type='AVSplit',**modelkwargs):
        """
        function to select model object and assert whether the model parameters are set up correctly
        Parameters:
        ----------
        model_type: str 

        Returns: 
        ---------
            np.ndarray
            log odds

        todo: redefine model when model contribution is assessed (i.e. fixedparam business)
        """
        if model_type=='AVSplit':
            model = AVSplit(**modelkwargs)

        assert (model.fixed_parameters.size==model.fixed_paramsValues.size),'recheck param no. for fixed Params & values '
        assert (np.setdiff1d(self.predictor_names,model.required_conditions).size==0), 'some of the required predictors have not been passed'
               
        # set up parameters and bounds        
        nFittableParams = model.required_parameters.size - np.sum(model.fixed_parameters) + self.n_neurons  # trying to add the regulariser        
        model.paramInit = [1] * nFittableParams
        nonNeuralFittables = model.required_parameters[~model.fixed_parameters[:model.required_parameters.size].astype('bool')]
        model.paramBounds = [(0,3) if m=='gamma' else (-50,50) for m in nonNeuralFittables]     
        if nFittableParams-nonNeuralFittables.size>0:
            [model.paramBounds.append((None,None)) for i in range(nFittableParams-nonNeuralFittables.size)]
        # replace the gamma bound ---


        return model
 
    def calculatepHat(self,conditions,betas):
        """
        calculate the probability of making each possible choice (i.e. [R L])
        """
        # grab all the parameters for the model
        params = self.model.get_all_params(betas) 
        logOdds = self.model.get_logOdds(conditions,params)
        pR = np.exp(logOdds) / (1 + np.exp(logOdds))
        pHat = np.array([1-pR,pR])    # because left choice=0 and thus it needs to be 0th index    
        return pHat    

    def init_data_for_LikeLihood(self,X,y):
        self.X = X
        self.y = y

    def get_Likelihood(self,betas): 
        """
        likelihood calculation that is typically used while fitting the model that minimises the likelihood
        """
        # this is what could be looped potentially given if there are several dims y becomes 2D & X becomes 3D
        # and then we just sum over the likelihoods for this we need a paramgenerator function called here        
        
        assert hasattr(self,'X'),'data input was not initialised correctly'
        alpha = 0 
        penalty = np.sum(np.abs(betas[self.model.required_parameters.size:]))*alpha;  # push away from the minimum with increasing by the penalty
        pHat_calculated = self.calculatepHat(self.X,betas) # the probability of each possible response 
        responseCalc = self.y # the actual response taken        
        # calculate how likely each of these choisen response was given the model
        logLik = -np.mean(np.log2(pHat_calculated[responseCalc.astype('int'),np.arange(pHat_calculated.shape[1])])) + penalty
        return logLik
    
    def get_Likelihood_per_trial(self,betas): 
        """
        likelihood calculation for each trial type, function used during evaluation (i.e. something that e.g. does not add the penalty when performing the evaluation)
        
        returns the -log2likelihood per trial
        """
        pHat_calculated = self.calculatepHat(self.X,betas) # the probability of each possible response 
        responseCalc = self.y
        logLiks_per_trial = -np.log2(pHat_calculated[responseCalc.astype('int'),np.arange(pHat_calculated.shape[1])])
        return logLiks_per_trial
        	
       
    def fit(self):
        """
        fit the model by minimising the logLikelihood
        i.e. the get_Likelihood function  
        todo: optimse parameters for search
        """
        # if the fitting has not been initialised with a dataset alrady ....
        if not hasattr(self,'X'):
            self.init_data_for_LikeLihood(self.conditions,self.choices)

        fittingObjective = lambda b: self.get_Likelihood(b)
        result = minimize(fittingObjective, self.model.paramInit, bounds=self.model.paramBounds)          
        self.model.LogLik = self.get_Likelihood(result.x)
        self.model.LogLik_per_trial = self.get_Likelihood_per_trial(result.x)
        self.model.paramFit = result.x
        self.model.allParams = self.model.get_all_params(result.x)
    
    def fitCV(self,**kwargs):

        sss = StratifiedShuffleSplit(**kwargs)
        X = self.conditions
        y = self.choices

        non_neuralX = X[:,:self.model.required_conditions.size]
        unique_conds = np.unique(non_neuralX,axis=0)

        fitted_params,params,logLiks,ll_per_cond = [],[],[],[]
        for train_index, test_index in sss.split(non_neuralX,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.init_data_for_LikeLihood(X_train,y_train)
            self.fit()
            
            fitted_params.append(self.model.paramFit[np.newaxis,:])
            params.append(self.model.allParams[np.newaxis,:])
            self.init_data_for_LikeLihood(X_test,y_test)
            logLiks.append(self.get_Likelihood(self.model.paramFit))

            self.init_data_for_LikeLihood(X_test,y_test)
            logLik_per_trial = self.get_Likelihood_per_trial(self.model.paramFit)

            # also calculate the logLikelihood per each trial condition
            non_neuralX_test = non_neuralX[test_index]
            ll_per_cond.append(np.array([np.mean(logLik_per_trial[(non_neuralX_test==c).all(axis=1)]) for c in unique_conds])[np.newaxis,:])

        # convert ll_per_cond_to_df
        # nanmean because if there was no trial in some splits we get a nan
        ll_per_cond = np.nanmean(np.concatenate(ll_per_cond),axis=0)
        ll_per_cond  = np.concatenate((unique_conds,ll_per_cond[:,np.newaxis]),axis=1)
        columns = self.model.required_conditions.tolist()
        columns.append('LogLik')
        ll_per_cond = pd.DataFrame(ll_per_cond,columns=columns)

        self.model.LogLik_per_condition = ll_per_cond
        self.model.LogLik=np.mean(logLiks)
        self.model.paramFit = np.mean(np.concatenate(fitted_params),axis=0)  
        self.model.allParams = np.mean(np.concatenate(params),axis=0)

    def visualise(self,**plotkwargs):
        """
        visualise the prediction of the log odds model, given visDiff & audDiff (default visualisation)
        """ 
        self.model.plot(parameters=self.model.allParams,conditions=self.conditions,choices=self.choices,**plotkwargs)
 
    def plotPrediction(self,**plotkwargs):
        self.model.plotPred(parameters=self.model.allParams,conditions=self.conditions,choices=self.choices,**plotkwargs)




def search_for_neural_predictors(rec,my_ROI='SCm',ll_thr = 0.005,**kwargs):
    """
    iterative search method to add neural predictors to the GLM from the 

    method: 
    
    Parameters:
    -----------
    rec: pd.Series
    my_ROI: str 
        Beryl acronym that identifies the ROI 
    event_type: str
        event to align to 
    ll_thr: 
        minimum log-likelihood decrease required for a neuron to be added
    
        
    Returns: 
    pd.df,list
    :the final matrix containing all variables including vis, aud, choice and neurons 
    :the ID of the neurons that have been searched..? 
    : logLikelihoods??

    """ 

    ev,spk,clusInfo,_,_ = simplify_recdat(rec,probe='probe')
    goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym==my_ROI)]._av_IDs.values
    trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,**kwargs)
    # iterative fitting for each nrn 
    nrn_IDs = [re.split('_',i)[1] for i in trials.columns if 'neuron' in i]

    non_neural = trials.iloc[:,:3]
    neural = trials.iloc[:,3:]
    glm = glmFit(non_neural,model_type='AVSplit',
                 fixed_parameters = [0,0,0,0,1,0],
                 fixed_paramValues=[1,1,1,1,0.7,0])
    glm.fitCV(n_splits=2,test_size=0.5)

    n_neurons = neural.shape[1]

    best_nrn,ll_best = [],[]
    ll_best = [glm.model.LogLik]
    for i in range(n_neurons):
        print('finding #%.0f best neuron' % i)
        if i==0:
            base_matrix = non_neural
            bleed_matrix = neural
        else:
            base_matrix = pd.concat((non_neural,neural.loc[:,best_nrn]),axis=1)
            leftover_nrn = np.setdiff1d(neural.columns.values,np.array(best_nrn))
            bleed_matrix = neural.loc[:,leftover_nrn]
            
        ll = []
        for idx,(neuronName,trial_activity) in enumerate(bleed_matrix.iteritems()):
            fittable = pd.concat((base_matrix,trial_activity),axis=1)
            neuralglm = glmFit(fittable,model_type='AVSplit',fixed_parameters = [0,0,0,0,1,0],fixed_paramValues = list(glm.model.allParams))
            neuralglm.fitCV(n_splits=2,test_size=0.5)

            ll_current = neuralglm.model.LogLik
            if np.isnan(ll_current) or np.isinf(ll_current):
                ll_current= 1000
            ll.append(ll_current)
        
        curr_best_ll = np.min(np.array(ll))
        if curr_best_ll>(ll_best[i]-ll_thr):
            print('the situation is not improving, you got to break...')
            break 

        ll_best.append(curr_best_ll)
        best_nrn.append(bleed_matrix.columns.values[np.argmin(np.array(ll))])

    final_matrix = pd.concat((non_neural,neural.loc[:,best_nrn]),axis=1)

    return final_matrix,goodclusIDs,best_nrn,ll_best

# maybe fore 
      

# %%
