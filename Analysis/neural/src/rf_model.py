import numpy as np
import scipy.optimize as opt
import itertools,re
import xarray as xr 
import matplotlib.pyplot as plt

from Analysis.neural.utils.spike_dat import get_binned_rasters,bin_spikes_pos_and_time, bin_mua_per_depth
from Admin.csv_queryExp import load_ephys_independent_probes
from Analysis.pyutils.plotting import off_axes

def get_start_params(myresp):
    """
    standard starting parameters for fitting -- best is to initialise around RF sizes that we expect 
    so hence sigma= 2 squares = 15 deg 

    """
    max_amplitude=np.max(myresp) 
    x0=np.argmax(np.mean(myresp,axis=0))
    y0=np.argmax(np.mean(myresp,axis=1))
    sigmaX=2#get_sigma(np.arange(0,myresp.shape[1],1),np.mean(myresp,axis=0))
    sigmaY=2#get_sigma(np.arange(0,myresp.shape[0],1),np.mean(myresp,axis=1))
    
    return np.array([max_amplitude,x0,sigmaX,y0,sigmaY])

def f_2D_gaussian(x,y,a,x0,sigmaX,y0,sigmaY):
    return a*np.exp(-((x-x0)**2/(2*sigmaX**2)+(y-y0)**2/(2*sigmaY**2)))

# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr += f_2D_gaussian(x, y, *args)
    return arr

def get_VE(actual,predicted):
    """
    calculating variance explained for any sort of array. 
    """
    actual_=np.ravel(actual)
    predicted_ = np.ravel(predicted)
    VE = 1-(np.var(actual_-predicted_)/np.var(actual_))

    return VE 

class rf_model(): 
    """
    this class is used to preprocess and fit visual receptive fields using sparseNoise protocols

    
    """
    def __init__(self,**csvkwargs): # maybe sparseNoise needs to go from init
        """
        initiate screen parameters and parameters for binning rasters

        """
        self.azimuth_idx=np.arange(0,36)
        self.elevation_idx=np.arange(0,10)
        self.azi_loc_screen,self.elevation_loc_screen=np.meshgrid(self.azimuth_idx,self.elevation_idx)
        self.screenpos = np.vstack((self.azi_loc_screen.ravel(), self.elevation_loc_screen.ravel()))  


        # initiate default parameters for rasters 
        self.raster_kwargs =  {
                'pre_time':0.2,
                'post_time':0.08, 
                'bin_size':0.01,
                'smoothing':0.025,
                'return_fr':True,
                'baseline_subtract': True, 
        }

        # initiate default parameters for calling the experiments

        self.ephys_dict =  {'spikes': 'all'} # load the ephys data
        self.other_dicts = {'events':{'_av_trials':'all'}} # load the events 

        if bool(csvkwargs) is True: 
            # if user gives parameters to call experiment, we load it
            self.load_single_session(**csvkwargs)

    def load_single_session(self,**csvkwargs):

        loader_kwargs = {
            'expDef':'sparseNoise',
            'checkSpikes': '1',  
            'ephys_dict': self.ephys_dict,
            'add_dict': self.other_dicts
            }

        loader_kwargs = {**loader_kwargs,**csvkwargs}
        recordings = load_ephys_independent_probes(**loader_kwargs)
        
        if len(recordings)>1: 
            print('numerous sessions are called ....')            
        else:
            rec = recordings.iloc[0]
            sn_info = rec.events._av_trials
            self.format_events(sn_info) 
            self.spikes = rec.probe.spikes

    def format_events(self,ev):
        """

        function to format events from PinkRigs to something more clear

        """
        self.azimuths = np.unique(ev.squareAzimuth)
        self.elevations = np.unique(ev.squareElevation)
        self.xy_pos = np.array([ev.squareAzimuth,ev.squareElevation]).T
        self.xy_times = ev.squareOnTimes
       

    def bin_spikes_per_loc(self,spikes,depth_corr_window_spacing=40,t_bin=0.01):
        """
        applying a temporal and spatial binning on the mua to fit mua/depth receptive fields. 

        """
        self.t_bin=t_bin
        self.binned_spikes_depths = bin_spikes_pos_and_time(spikes,depth_corr_window_spacing=depth_corr_window_spacing,spike_binning_t=self.t_bin)

    def get_response_binned(self,sel_depth_binned,t_before=0.2,t_after=0.06,t_delay=0.02):
        """
        some getting response after you have already binned the entire data 
        Parameters: 
        -----------
        sel_depth_binned: np.ndarray 
            1D digitised array of spikes, i.e. only applicable to depth binned responses

        """
        before_ix=int(t_before/self.t_bin)
        after_ix=int(t_after/self.t_bin)
        delay_ix=int(t_delay/self.t_bin)
        
        sNstartix=np.ceil(self.xy_times*(1/self.t_bin)).astype("int")
        response=np.zeros((self.azimuths.size,self.elevations.size))

        for xct,my_x in enumerate(self.azimuths): 
            for yct,my_y in enumerate(self.elevations): 
                sq=[my_x,my_y]
                ix=np.where((self.xy_pos==sq).all(axis=1))[0]
                
                nrlix=np.ravel(sNstartix[ix])

                resp_sq=np.zeros((nrlix.shape[0],before_ix+after_ix))
                for i in range(before_ix+after_ix):
                    ixs=nrlix-before_ix+i
                    resp_sq[:,i]=sel_depth_binned[ixs]

                blmean=(resp_sq[:,:before_ix].sum(axis=1))/t_before
                respmean=(resp_sq[:,(before_ix+delay_ix):].sum(axis=1))/(t_after-t_delay)
                response[xct,yct]=np.abs((respmean-blmean).mean(axis=0)/blmean.std(axis=0))
                #mypred,ve,fitted_params=self.fit_predict(response.T)

        return response.T
    
    def get_response_spike_triggered():
        pass
    
    def get_response_stim_triggered(self,mode='per-neuron',shuffle_seed=None,selected_ids=None,delay_for_vistrig=0.02,cv_split = 2):
        """
        function to get stimulus triggered averages for the sparseNoise protocol 
        Parameters: 
        -----------
        mode: str
            'per-neuron' : response calculated per clusterIDs
            'per-depth': response calculated from mua per location -- to be implemented
        shuffle_seed: None/float
            wherther to shuffle the square position that was onsetted (can be slow, because I recalculate averages, so don't do many)        
        selected_ids: None/list
            cluster IDs of neurons where response calculation is requested
        delay_for_vistrig: float
            time (in sec) to take the response fro relative to actual square onset (meant to account for the fact that visual responses are rather slow)
        cv_split: float 
            no. of cross-validation sets

        Returns: xarray.DataArray
            of response averages, per parameters (clusterID, screen location,cv_split) 
        """
        if mode=='per-neuron':
            all_cluster_ids = self.spikes.clusters
        elif mode == 'per-depth': 
            spikes_depths = bin_mua_per_depth(self.spikes,depth_spacing=60)            
            all_cluster_ids = spikes_depths.shank_depth_ids
            
        if selected_ids is None:
            requsted_cluster_ids = np.unique(all_cluster_ids)
        else:
            requsted_cluster_ids = selected_ids

        if shuffle_seed is None: 
            xy_pos = self.xy_pos
        else:
            np.random.seed(shuffle_seed) 
            xy_pos =  np.random.permutation(self.xy_pos)

        response_averages = np.zeros((requsted_cluster_ids.size,self.elevations.size,self.azimuths.size,cv_split))
        for xct,my_x in enumerate(self.azimuths): 
            for yct,my_y in enumerate(self.elevations): 
                sq=[my_x,my_y]
                ix=np.where((xy_pos==sq).all(axis=1))[0]
                t_on_square = self.xy_times[ix]

                # create cross_val_sets
                np.random.seed(0)
                t_on_square = np.random.permutation(t_on_square)
                split_edges = list(itertools.accumulate(itertools.repeat(int(t_on_square.size/cv_split),cv_split)))
                split_edges.insert(0,0)

                for cv in range(cv_split):
                    onsets = t_on_square[split_edges[cv]:split_edges[cv+1]]
                    r = get_binned_rasters(self.spikes.times,all_cluster_ids,requsted_cluster_ids,onsets,**self.raster_kwargs)
                    stim = np.mean(np.mean(r.rasters[:,:,r.tscale>=delay_for_vistrig],axis=0),axis=-1) # average response in time and across trials
                    response_averages[:,yct,xct,cv] = stim 

        responses=xr.DataArray(response_averages,
                                dims=('neuronID','elevations','azimuths','cv_number'),
                                coords={'neuronID':requsted_cluster_ids,
                                        'elevations':self.elevations,
                                        'azimuths':self.azimuths,
                                        'cv_number':range(cv_split)}) 
        
        return responses
        
    def optimise_2Dgauss(self,response):
        """
        response: elevation x azimuth (10,36)

        """
        p0 = get_start_params(response)
        try:
            popt, _ = opt.curve_fit(_gaussian, self.screenpos, response.ravel(), p0)
            fitted_params=popt
            # prediction
        except RuntimeError: # this is when the curve fit could not manage to fit a gaussian and it possibly does not exist
                fitted_params=np.empty(p0.shape)*np.nan
        
        return fitted_params
    

    def fit(self,**response_kwargs):
        """
        function to fit and predict responses for several mua/neurons
        have to be on cv_number >1  
        """

        self.responses = self.get_response_stim_triggered(**response_kwargs)
        # optomise curve fit and store values
        fit_params = [self.optimise_2Dgauss(self.responses.sel(cv_number=0,neuronID=nrn).values)[np.newaxis,:] for nrn in self.responses.neuronID.values]        
        self.fit_params = np.concatenate(fit_params)

    def get_rf_degs_from_fit(self): 
        pref_azimuth = (self.fit_params[:,1])*7.5-135
        azimuth_sigma = (self.fit_params[:,2])*7.5
        pref_elevation = 75-7.5*(self.fit_params[:,3])-41.25 
        elevation_sigma = (self.fit_params[:,4])*7.5 
        return pref_azimuth,pref_elevation,azimuth_sigma,elevation_sigma

    def predict(self):
        # predict 
        predictions = [f_2D_gaussian(self.azi_loc_screen,self.elevation_loc_screen,*p)[np.newaxis,:,:] for p in self.fit_params]

        self.predictions = xr.DataArray(np.concatenate(predictions),
                                dims=('neuronID','elevations','azimuths'),
                                coords={'neuronID':self.responses.neuronID.values,
                                        'elevations':self.elevations,
                                        'azimuths':self.azimuths}) 
        
        
    def evaluate(self):
        """calculate variance explained by each 2D gauss fitting"""
        VE = [np.array([get_VE(
                self.responses.sel(neuronID=nrn,cv_number=cv),self.predictions.sel(neuronID=nrn)) for nrn in self.responses.neuronID]
                )[:,np.newaxis]  
                for cv in self.responses.cv_number] 
        
        self.score=xr.DataArray(np.concatenate(VE,axis=1),
                                dims=('neuronID','cv_number'),
                                coords={'neuronID':self.responses.neuronID.values,
                                        'cv_number':self.responses.cv_number.values}) 
        
        

    def fit_evaluate(self,**response_kwargs):

        self.fit(**response_kwargs)
        self.predict() 
        self.evaluate() 


    def plot_fit(self,ID='0',**response_kwargs):
        """
        can be modified for optional plotting, but for now plots the prediction and data for each cross-val set 
        set 0 is always the training set. 

        ID: float/str/list
            if str, we assume we are dealing with mua plotting of a shank

        """
        # recalculate predicitons if needed
        if not hasattr(self,'predictions'):
            self.fit_evaluate(**response_kwargs)
        if type(ID)==str:
            ID+= '-'
            all_IDs = self.predictions.neuronID.values
            shankIDs = [re.split('-',nrn)[0] for nrn in all_IDs]
            depths = np.array([float(re.split('-',nrn)[1]) for nrn in all_IDs])
            is_called = [(ID in nrn) for nrn in all_IDs]
            ID = all_IDs[is_called]
            sorted_idx = np.argsort(depths[is_called])
            ID = ID[sorted_idx]

        elif type(ID)==int: 
            ID = [ID]

        fig,ax = plt.subplots(len(ID),self.responses.cv_number.size+1,sharex=True,sharey=True,figsize=(20,10))        
        if ax.ndim==1:
            ax = ax[np.newaxis,:]
        # plot the prediction
        for idx,cID in enumerate(ID):
            pred_ =self.predictions.sel(neuronID=cID).values
            vmax = vmax=np.max(np.abs(pred_))/2
            im = ax[idx,0].imshow(pred_,vmin=-vmax,vmax=vmax,cmap='coolwarm')
            if (type(cID)==np.str_):
                if len(ID)<10:
                    ax[idx,0].set_title('neuron %s' % cID)
            else:
                ax[idx,0].set_title('neuron %.0d' % cID)

            ax[idx,0].set_xticks(np.round(self.azimuths[self.azimuth_idx[::10]],0))
            ax[idx,0].set_yticks(np.round(self.elevations[self.elevation_idx[::10]],0))
            # then plot the data
            for cv in self.responses.cv_number:
                resp_ =  self.responses.sel(neuronID=cID,cv_number=cv).values
                vmax = vmax=np.max(np.abs(resp_))/2
                im = ax[idx,cv+1].imshow(resp_,vmin=-vmax,vmax=vmax,cmap='coolwarm')
                curr_score =  self.score.sel(neuronID=cID,cv_number=cv).values
                if ~np.isnan(curr_score):
                    if len(ID)<10:
                        ax[idx,cv+1].set_title(
                            'cv set %.0d, VE: %.2f' % (cv,curr_score)
                        )                
                off_axes(ax[idx,cv+1])
        
        fig.colorbar(im, ax=ax, shrink=0.3)

    def get_significant_rfs(self,n_shuffles=20,**response_kwargs):
        print('evaluating significance. This might take a while.')
        shuffle_scores = []
        for n in range(n_shuffles):
            self.fit_evaluate(shuffle_seed = n, **response_kwargs)
            shuffle_scores.append(self.score.sel(cv_number=1).values[np.newaxis,:]) # cv score on the test set 

        shuffle_scores = np.concatenate(shuffle_scores).T

        self.fit_evaluate(shuffle_seed = None, **response_kwargs)
        actual_score = self.score.sel(cv_number=1).values
        p_value = np.sum(shuffle_scores>np.tile(actual_score[:,np.newaxis],n_shuffles),axis=1)/n_shuffles
        self.is_significant = p_value<0.05

        return  self.is_significant


    def fit_mua(self,response):
        """
        fitting procedure for response 10x36 matrix (stim triggered/spike triggered average)
        fast? fitting procedure I used to use to fit RFs. Probably the cross-validation procedure is much better
        and I should overwrite my RF search function using that one eventually. 
        """

        # if the response is also 0/inf/nan the fit will error and that cannot be fitted anyway.

        if np.isnan(response).any():
            fitted_params=np.nan
        else:
            p0 = get_start_params(response)
            try:
                popt, _ = opt.curve_fit(_gaussian, self.screenpos, response.ravel(), p0)
                fitted_params=popt
                # prediction
                pred=f_2D_gaussian(self.azi_loc_screen,self.elevation_loc_screen,*fitted_params)
                ve = get_VE(response,pred) # predict the trainng set
                # probably not a real fit  # really should assess better
                if ve<.3:
                    fitted_params=np.nan
            except RuntimeError: # this is when the curve fit could not manage to fit a gaussian after 1200 iterations
                    fitted_params=np.nan
                
        return fitted_params
    
    