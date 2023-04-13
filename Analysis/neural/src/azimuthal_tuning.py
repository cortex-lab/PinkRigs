
import itertools, re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from Admin.csv_queryExp import load_ephys_independent_probes,Bunch
from Analysis.pyutils.ev_dat import postactive
from Analysis.pyutils.model_funcs import get_VE
from Analysis.neural.utils.spike_dat import get_binned_rasters

# load data
import sklearn
import scipy

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def genericTune(x,ybot,ytop,x0,sigmaL,sigmaR):
    """
    1D Gaussian where sigma towards the left vs the right is allowed to vary 
    Parameters: 
    -----------
    x: np.ndarray
    ybot: float
        baseline of the Gaussian
    ytop: float 
        determines the magnitude of the Gaussian
    x0: float 
        center
    sigmaL: float
        sigma towards left
    sigmaR: float
        sigma towards right

    """
    left = ybot+(ytop-ybot)*np.exp(-((x-x0)**2/(2*sigmaL**2)))
    left[(x>=x0)] = 0 # sum the half gaussians
    right = ybot+(ytop-ybot)*np.exp(-((x-x0)**2/(2*sigmaR**2)))
    right[(x<x0)] = 0 
    return (left+right)

def init_params_fitGeneric(x,y):
    """
    function to initilaise parameters 
    """
    ybot = np.min(y)
    ytop = np.max(y)
    x0 = x[np.argmax(y)]
    d = (ytop-ybot)/(ytop+ybot)
    sigmaL = 10/d 
    sigmaR = 10/d

    return [ybot,ytop,x0,sigmaL,sigmaR]

def upsample(x,y,upfactor = 100,**kwargs):
    """
    upsample values using scipy.interpolate1d

    Parameters: 
    x: numpy ndarray
    y: numpy ndarray 
    upfactor: float 
        factor by which we upsample x
    
    Returns: np.ndarrays  
    --------
    x_
    y_ 

    """
    interp_func = scipy.interpolate.interp1d(x,y,**kwargs)
    x_ = np.linspace(np.min(x),np.max(x),x.size*upfactor)
    y_ = interp_func(x_)
    return x_,y_

def fitGeneric(x,y,upfactor= None,p0=None):   
    """
    p0:None/list 
        parameters for genericTime 
    """ 
    if upfactor is not None: 
        x,y = upsample(x,y,upfactor=upfactor)

    if p0 is None: 
        p0 = init_params_fitGeneric(x,y) 


    try:
        fitted_params, _ = scipy.optimize.curve_fit(genericTune,x,y,p0=p0)
    except RuntimeError:         
        fitted_params = np.empty(len(p0))*np.nan


    return fitted_params


def svd_across_azimuths(r): 
    """
    helper function to get the tuning curve via the svd method for each neuron 
    details: 1) svd across all responses (independent of azimuth)
             2) get the weight of every trial on the 1st PC
             3) average and take the absolute value of weights

    Parameters: 
    -----------
    r: np.ndarray
        azimuth x trials x time raster matrix

    Returns: np.ndarray,np.ndarray
        weights per trial  ... and 
        the mean weights of PC1 along the azimuth axis (1D array)

    """
    r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
    [u,_,_] = np.linalg.svd(r_,full_matrices=False)
    weight_per_trial = np.reshape(u[:,0],(r.shape[0],r.shape[1])) # weights of every trial on PC1
    weight_per_trial = np.abs(weight_per_trial)    
    average_weight = weight_per_trial.mean(axis=1)

    return weight_per_trial, average_weight[np.newaxis,:]

def get_tuning_only(tuning_curves):
    """
    helper function to handle the tuning curves dataframe

    """

    isgood = ~tuning_curves.columns.isin(['score','cv_number','preferred_tuning'])
    tc = tuning_curves.loc[:,isgood]

    return tc

class azimuthal_tuning():
    def __init__(self,rec_info):
        self.raster_kwargs = {
                'pre_time':0.6,
                'post_time':0.2, 
                'bin_size':0.01,
                'smoothing':0.025,
                'return_fr':True,
                'baseline_subtract': True, 
        }
        self.load(rec_info)

    def load(self,rec_info):

        ephys_dict =  {'spikes': ['times', 'clusters']}
        other_ = {'events': {'_av_trials': 'table'}}

        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**rec_info)
        if recordings.shape[0] == 1:            
            recordings =  recordings.iloc[0]
        else:
            print('recordings are ambiguously defined. Please recall.')
        events,self.spikes = recordings.events._av_trials,recordings.probe.spikes
        _,self.vis,self.aud,self.ms = postactive(events)

    def get_rasters_perAzi(self,contrast = None, spl = None, which = 'vis', subselect_neurons = None):
        """
        get rasters per azimuth for the loaded data

        Parameters: 
        ----------
        contrast: numpy.ndarray
        spl: numpy.ndarray
        which: str
            'vis' or 'aud'
        subselect_neurons: list
            cluster_ids subselection not implemented. 
        cv_split: float
            no of splits for cross_validation
        
        Returns:
        --------
            :Bunch
        .dat: np.ndarray
            azimuths x trials x neurons x timebins     
        .azimuthIDs: list of str
            which azimuths we actually included.         
        """
        if not contrast: 
            contrast = np.max(self.vis.contrast.values)

        if not spl:
            spl = np.max(self.aud.SPL.values)
        else: 
            spl = np.min(self.aud.SPL.values)
            
        if not subselect_neurons: 
            self.clus_ids = np.unique(self.spikes.clusters)    
        else:
            pass

        if 'vis' in which:
            azimuth_options = sorted(self.vis.azimuths.values)
            onset_matrix = self.vis.sel(contrast=contrast,timeID='ontimes')

        elif 'aud' in which:
            azimuth_options = sorted(self.aud.azimuths.values)
            onset_matrix = self.aud.sel(SPL=spl,timeID='ontimes')

        elif 'coherent' in which:
            azimuth_options = sorted(self.ms.congruent_azimuths[0,:])
            onset_matrix = self.ms.sel(contrast=contrast,SPL=spl,timeID = 'visontimes')

                           
        azimuth_times_dict = {}
        for azimuth in azimuth_options:
            if 'coherent' in which:
                azimuth_times_dict[('%s_%.0f' % (which,azimuth))] = onset_matrix.sel(
                    visazimuths = azimuth,
                    audazimuths = azimuth,
                ).values
            else: 
                azimuth_times_dict[('%s_%.0f' % (which,azimuth))] = onset_matrix.sel(
                    azimuths = azimuth
                ).values

        response_rasters = []
        for k in azimuth_times_dict.keys():
            t_on = azimuth_times_dict[k]
            r = get_binned_rasters(self.spikes.times,self.spikes.clusters,self.clus_ids,t_on,**self.raster_kwargs)
            # so we need to break here so that we perorm getting binned rasters only once...
            stim = r.rasters[:,:,r.tscale>=0]
            response_rasters.append(stim[np.newaxis,:,:,:])       
        
        self.response_rasters_per_azimuth = Bunch({
            'dat':np.concatenate(response_rasters,axis=0), 
            'azimuthIDs':list(azimuth_times_dict.keys())
            })
        
        self.azimuths = np.array([int(re.split('_',azi)[-1]) for azi in self.response_rasters_per_azimuth.azimuthIDs])
        return self.response_rasters_per_azimuth

    def plot_response_per_azimuth(self,neuronID=1,which='psth'):
        """
        plot the response at each azimuth for give n neuron 

        Parameters:
        ----------- 
        neuronID: float 
        which: str
            'p' if psth
            'r' if raster
        """
        neuron_idx = np.where(self.clus_ids==neuronID)[0][0]

        response = self.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
        fig,ax = plt.subplots(1,response.shape[0],figsize=(15,3),sharex=True,sharey=True)
        for i,r in enumerate(response):
            if 'r' in which:
                ax[i].imshow(r,cmap='Greys')
            elif 'p' in which:
                ax[i].plot(r.mean(axis=0))

            ax[i].set_title(self.response_rasters_per_azimuth.azimuthIDs[i]) 

        fig.suptitle('neuron %.0d' % neuronID)           

    def get_tuning_curves(self,rasters = None,cv_split=2,azimuth_shuffle_seed=None,metric='abs-max'):
        """
        Parameters: 
        -----------
        rasters: np.ndarray 
            azimuths x trials x neurons x timebins 
        cv_split: float
            no. of cross-validation sets to divide the data to 
        azimuth_shuffle_seed: None/float 
            None: no shuffle 
            float: seed number of shuffling the azimuth labels prior to arraning responses into a matrix per azimuth
        mode: str
            label on how to calculate the response metric at a given azimuth 
            options: 
            abs-max: takes the abs of the mean across trials, and then taking the max of that
            svd: svd across all responses, and take average PC1 weight for a given azimuth
        """
        if not rasters:
            rasters = self.response_rasters_per_azimuth.dat
            azimuths = self.response_rasters_per_azimuth.azimuthIDs
        else:
            azimuths = range(rasters.shape[0])

        azimuth_inds = range(rasters.shape[0])

        # shuffle traces across azimuths
        if azimuth_shuffle_seed: 
            collated_rasters = rasters.reshape(rasters.shape[0]*rasters.shape[1],rasters.shape[2],rasters.shape[3])
            # permute along 1st index 
            np.random.seed(azimuth_shuffle_seed)
            collated_rasters = np.random.permutation(collated_rasters)
            rasters = collated_rasters.reshape(rasters.shape[0],rasters.shape[1],rasters.shape[2],rasters.shape[3])
        
        # shuffle trial indices
        trials_idx = self.aud.trials.values
        np.random.seed(0)
        trials_idx = np.random.permutation(trials_idx)
        split_edges = list(itertools.accumulate(itertools.repeat(int(trials_idx.size/cv_split),cv_split)))
        split_edges.insert(0,0)

        tuning_curves = []
        for cv in range(cv_split):
            responses = {}
            curr_idxs = trials_idx[split_edges[cv]:split_edges[cv+1]] # trial indices of current set 
            stim_all_azimuth = rasters[:,curr_idxs,:,:] # azimuth x trials (cv_set) x neuron x time

            if 'svd' in metric: 
                n_neurons = stim_all_azimuth.shape[2]
                _ , responses = zip(*[svd_across_azimuths(stim_all_azimuth[:,:,n,:]) for n in range(n_neurons)])
                curr_tuning_curves = pd.DataFrame(
                    np.concatenate(responses),
                    columns=azimuths
                )

            elif 'abs-max' in metric: 
                for k in azimuth_inds:
                    stim = stim_all_azimuth[k,:,:,:]                                  
                    Rmax_stim = np.max(np.abs(stim.mean(axis=0)),axis=1)
                    responses[azimuths[k]] = Rmax_stim

                curr_tuning_curves = pd.DataFrame.from_dict(responses)  

            # interpolate the tunining curves

            preferred_tuning_idx = np.argmax(curr_tuning_curves.to_numpy(),axis=1)

            curr_tuning_curves[('preferred_tuning')] = [re.split('_',curr_tuning_curves.columns[i])[-1] for i in preferred_tuning_idx]
            curr_tuning_curves['cv_number'] = cv
            curr_tuning_curves = curr_tuning_curves.set_index(self.clus_ids,drop=True)
            tuning_curves.append(curr_tuning_curves)
        
        tuning_curves = pd.concat(tuning_curves)

        return tuning_curves
    

    
    def fit_tuning_curve(self,tuning_curves=None,**kwargs):
        """

        Parameters: 
        -----------
        tuning_curves: pd.DataFrame 

        """
        if tuning_curves is None:            
            tuning_curves = self.get_tuning_curves(cv_split=1,**kwargs)   

        # convert tuning curves df to numpy array and fit the training set 
        tc_train = get_tuning_only(tuning_curves[tuning_curves.cv_number==0]).values
        azimuths = self.azimuths
        # fit each neuron 
        p0 = np.concatenate([fitGeneric(azimuths,n,upfactor=100)[np.newaxis,:] for n in tc_train])
        self.tc_params = pd.DataFrame(data=p0,columns=['ybot','ytop','x0','sigmaL','sigmaR'],index = self.clus_ids)

        # evaluate on the 2nd half

    def predict_tuning_curve(self,azimuths=None):      

        if azimuths is None: 
            azimuths = self.azimuths

        self.predictions = np.concatenate([genericTune(azimuths,*p)[np.newaxis,:] for _,p in self.tc_params.iterrows()])

    def evaluate_tuning_curve(self,tuning_curves):
        self.predict_tuning_curve()
        cv_numbers = tuning_curves.cv_number.values
        
        score=[]
        for cv_number in np.unique(cv_numbers):
            tc = get_tuning_only(tuning_curves[tuning_curves.cv_number==cv_number]).values
            score.append([get_VE(actual,predicted) for actual,predicted in zip(tc,self.predictions)])
        score=np.concatenate(score)
        return score



    def fit_evaluate(self,**kwargs):
        tuning_curves = self.get_tuning_curves(**kwargs)   
        self.fit_tuning_curve(tuning_curves=tuning_curves)
        score = self.evaluate_tuning_curve(tuning_curves)      
        tuning_curves['score'] = score
        return tuning_curves

    def plot_tuning_curves(self,tuning_curves=None,metric='abs-max',neuronID=1,plot_train=True,plot_test=True,plot_pred=True): 
        """
        function to plot the tuning curve for a given neuron
        """
        if tuning_curves is None:            
            tuning_curves = self.get_tuning_curves(cv_split=2) 

        neuron_idx = np.where(self.clus_ids==neuronID)[0][0]
        # get azimuth values at which we are making calculations 

        azimuths = self.azimuths
        stim = self.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]

        n_azimuths = self.response_rasters_per_azimuth.dat.shape[0]
        n_trials = self.response_rasters_per_azimuth.dat.shape[1]

        # for each trial get the max across time    # this to be modified based on how I take the trials
        if 'abs-max' in metric:
            stim_per_trial = np.max(np.abs(stim),axis=2)
        elif 'svd' in metric: 
            stim_per_trial,_ = svd_across_azimuths(stim)

        figure,ax = plt.subplots(1,1,figsize=(6,7))
        for i in range(n_azimuths):
            ax.plot(np.ones((n_trials,1))*azimuths[i],stim_per_trial[i,:],'ko')


        titlestring = 'neuron %.0d' % neuronID
        if plot_train:
            tc_train = get_tuning_only(tuning_curves[tuning_curves.cv_number==0]).iloc[neuron_idx].values
            ax.plot(azimuths,tc_train,'b')
        
        if plot_test: 
            tc_test = get_tuning_only(tuning_curves[tuning_curves.cv_number==1]).iloc[neuron_idx].values
            ax.plot(azimuths,tc_test,'m')

        if plot_pred: 
            azimuths_upped = np.linspace(np.min(azimuths),np.max(azimuths),azimuths.size*100)
            p = self.tc_params.iloc[neuron_idx].values
            t_pred = genericTune(azimuths_upped,*p)
            ax.plot(azimuths_upped,t_pred,'k')

            if plot_train: 
                ve_string = ' VE,train = %.2f' % tuning_curves[tuning_curves.cv_number==0].iloc[neuron_idx].score
                titlestring += ve_string
            
            if plot_test: 
                ve_string = ' VE,test = %.2f' % tuning_curves[tuning_curves.cv_number==1].iloc[neuron_idx].score
                titlestring += ve_string

        
        ax.set_title(titlestring)

                


        
    


    def get_selectivity(self,**kwargs):
        """
        function to get cross-validated selectivity from responses at various azimuthal tunings

        Procedure: 
        ----------


        """
        tuning_curves = self.get_tuning_curves(cv_split=2,**kwargs)     

        tuning_type = tuning_curves.columns[0][:3]

        keys = [c for c in tuning_curves.columns if tuning_type in c and 'preferred' not in c]              

        train = tuning_curves[(tuning_curves.cv_number==0)]
        train = train[keys]

        test= tuning_curves[(tuning_curves.cv_number==1)]
        test = test[keys]

        max_loc = np.argmax(train.values,axis=1)
        min_loc = np.argmin(train.values,axis=1)

        max_test = test.values[range(test.shape[0]),max_loc]
        min_test = test.values[range(test.shape[0]),min_loc]

        selectivity = (max_test-min_test)/(max_test+min_test)

        return selectivity,tuning_curves[(tuning_curves.cv_number==0)].preferred_tuning.values.astype('float')
    
    def calculate_significant_selectivity(self,n_shuffles=100,p_threshold=0.01):
        if 1/n_shuffles>p_threshold:
            print('not enough shuffles for this p threshold')
                
        self.selectivity,self.preferred_tuning = self.get_selectivity(azimuth_shuffle_seed=None)
        
        s_shuffled,_ = zip(*[self.get_selectivity(azimuth_shuffle_seed=shuffle_idx) for shuffle_idx in range(n_shuffles)])
        s_shuffled = [s[np.newaxis,:] for s in s_shuffled]
        self.selectivity_shuffle_dist = np.concatenate(s_shuffled,axis=0)
        # calculate p value
        selectivity_ =np.tile(self.selectivity,(n_shuffles,1))
        p_val = (self.selectivity_shuffle_dist>selectivity_).sum(axis=0)/n_shuffles

        is_selective = p_val < p_threshold
        self.is_selective = is_selective
        return is_selective,self.preferred_tuning
            
    def plot_selectivity_distribution(self,clusID):
        cidx = np.where(self.clus_ids==clusID)[0][0]
        _,ax = plt.subplots(1,1)
        ax.hist(self.selectivity[:,cidx])
        ax.axvline(self.selectivity_shuffle_dist[cidx])
        ax.set_title(self.preferred_tuning[cidx])

    def get_enhancement_index(self,at_azimuth=None):
        """
        function to calculate traditional enhancement index at preferred azimuth 
        taken from Stanford et al. 2005 
        Method in brief:
        take every combination of unsory response trial additions 
        then resample that distribution with (skitlearn bootstrap)
        then get the mean and the std of the additive distribution 

        EI = (ms_mean-additive_mean)/additive_std

        Parameters: 
        -----------
        at_azimuth: np.ndarray
            estimated preferred azimuth for each cell.

        """
        if at_azimuth is None:
            at_azimuth = self.preferred_tuning

        vis = self.get_rasters_perAzi(which = 'vis')
        aud = self.get_rasters_perAzi(which = 'aud')
        ms  = self.get_rasters_perAzi(which = 'coherent')

        # get averages over time 
        vis.dat = vis.dat.sum(axis=3)
        aud.dat = aud.dat.sum(axis=3)
        ms.dat = ms.dat.sum(axis=3)

        # get only the ones at preferred index
        azimuths_vis = np.array([re.split('_',a)[-1] for a in vis.azimuthIDs]).astype('int')
        azimuths_aud = np.array([re.split('_',a)[-1] for a in aud.azimuthIDs]).astype('int')

        if (azimuths_vis==azimuths_aud).all():
            azimuths = azimuths_vis
        else:
            print('vis and aud azimuths are not the same, so indexing would be incorrect.')

        at_azimuth_idx = np.array([np.where(azimuths==preferred_nrn)[0][0] for preferred_nrn in at_azimuth])

        vis_trials = np.concatenate([vis.dat[idx,:,nrn][:,np.newaxis] for nrn,idx in enumerate(at_azimuth_idx)],axis=1)
        aud_trials = np.concatenate([aud.dat[idx,:,nrn][:,np.newaxis] for nrn,idx in enumerate(at_azimuth_idx)],axis=1) 
        ms_trials = np.concatenate([ms.dat[idx,:,nrn][:,np.newaxis] for nrn,idx in enumerate(at_azimuth_idx)],axis=1) 

        # get the summed distibution across trials
        additive_samples = np.concatenate(
            [np.ravel(vis_trials[:,nrn]+aud_trials[:,nrn][:,np.newaxis])[:,np.newaxis]
                      for nrn in range(vis_trials.shape[1])],
                      axis=1) 

        additive_dist_resampled = np.concatenate(
            [sklearn.utils.resample(additive_samples[:,s],replace=True,n_samples=10000)[:,np.newaxis] 
             for s in range(at_azimuth_idx.size)],
             axis=1
             )

        additive_mean = additive_dist_resampled.mean(axis=0)
        additive_std = additive_dist_resampled.std(axis=0)

        # get mulitsensory mean
        ms_mean = ms_trials.mean(axis=0)                
        # get enhancement index
        enhancement_index = (ms_mean-additive_mean)/additive_std

        return enhancement_index





            


