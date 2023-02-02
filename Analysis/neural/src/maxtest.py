
"""
This code is the implmentation of maxtest 
used in Coen, Sit et al., 2021
Maxtest is a response sigificance test that is agnostic to the duration of response, 
and thus allows transient auditory responses (of 1 spike/trial)to be detected in the same 
time window as longer visual responses. 

Methodology: 

1. bin, smooth
2. take a trimmed PSTH (whereby 10% of the largest and smallest amplitude responses 
were discarded) # maybe don't do that (?)
3. bl subtract 
4. take the maximum of the absolute value (to account for inhibition) of the response (Rmax). 
5. Test statistic was subtracting Rmax on blank trials from Rmax on stimulus trials. 
6. Get null distribution by shuffling blank and stimulus trials % and this is by design a right-sided permutation test
7. get p-value (Bonferroni correct if necessary). 

"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Analysis.neural.utils.spike_dat import get_binned_rasters
from Analysis.neural.utils.ev_dat import postactive
from Analysis.neural.utils.plotting import off_axes

from Admin.csv_queryExp import load_data, simplify_recdat

# get all the unique trial types 

def get_test_statistic(test_raster,blank_raster,permute_seed = None):
    # check if trial sizes are the same, if not subsample
    n_trials_test = test_raster.shape[0]
    n_trials_blank = blank_raster.shape[0]

    if n_trials_test>n_trials_blank:
        test_raster = test_raster[:n_trials_blank,:,:]
    elif n_trials_blank>n_trials_test: 
        blank_raster = blank_raster[:n_trials_test,:,:]

    rasters = np.concatenate((test_raster,blank_raster),axis=0)
    n_trials = rasters.shape[0]
    idx = np.arange(n_trials)
    if permute_seed:
        np.random.seed(permute_seed) 
        idx = np.random.permutation(idx)

    # split 
    s_p = int(n_trials/2)
    stim = rasters[idx[:s_p],:,:]
    blank = rasters[idx[s_p:],:,:]

    Rmax_stim = np.max(np.abs(stim.mean(axis=0)),axis=1)
    Rmax_blank = np.max(np.abs(blank.mean(axis=0)),axis=1)
    
    return Rmax_stim-Rmax_blank

class maxtest(): 
    def __init__(self):

        self.raster_kwargs = { 
                'pre_time':0.6,
                'post_time':0.2, 
                'bin_size':0.01,
                'smoothing':0.025,
                'return_fr':True,
                'baseline_subtract': True, 
            }
            
    def load_and_format_data(self,probe='probe0',**kwargs):
        data_dict = {
            'events':{'_av_trials':'table'},
            probe:{'spikes':['times','clusters']}
            }

        recordings = load_data(data_name_dict = data_dict,**kwargs)   
        events,spikes,_,_ = simplify_recdat(recordings.iloc[0])
        blanks,vis,aud,_ = postactive(events)
        event_dict = {}
        # to redo the previous tests
        checked_spl = np.max(aud.SPL.values)
        for (azimuth,d_power) in itertools.product(aud.azimuths.values,[checked_spl]):
            stub = 'aud_azimuth_%.0d_dpower_%.2f' % (azimuth,d_power)
            event_dict[stub] = aud.sel(azimuths=azimuth,SPL=d_power,timeID='ontimes').values
        for (azimuth,d_power) in itertools.product(vis.azimuths.values,vis.contrast.values):
            stub = 'vis_azimuth_%.0d_dpower_%.2f' % (azimuth,d_power)
            event_dict[stub] = vis.sel(azimuths=azimuth,contrast=d_power,timeID='ontimes').values 

        self.event_times_dict = event_dict 
        self.spikes = spikes
        self.blank_times = blanks.sel(timeID='ontimes').values 



    def run(self,spikes=None,event_times_dict=None,blank_times=None,subselect_neurons=None,plotting=False,savepath = None,n_shuffles=2000):
        """
        Parameters:
        -----------
        spikes: Bunch
        event_times_dict: dict
        blank_times: numpy.ndarray
        subselect_neurons: list
        n_shuffles: float

        plotting: bool, only allowed if subselect neuron is on.
            not recommended with more than 7-8 nrns and conditions. 
        
        Returns:
        -------
            : pd.DataFrame
        """

        if not spikes:
            spikes = self.spikes
        if not event_times_dict: 
            event_times_dict = self.event_times_dict
        if not blank_times:
            blank_times = self.blank_times

        if not subselect_neurons:
            clus_ids = np.unique(spikes.clusters)    
        else:
            clus_ids = np.array(subselect_neurons)

        blank = get_binned_rasters(
            spikes.times,spikes.clusters,
            clus_ids,blank_times,
            **self.raster_kwargs
            )
        blank = blank.rasters[:,:,blank.tscale>=0]

        p_value_per_event=[]

        if not subselect_neurons:
            plotting = False 

        if plotting:
            n_neurons = len(subselect_neurons)
            n_events = len(list(event_times_dict.keys()))
            fig,ax = plt.subplots(n_neurons,n_events,figsize=(2*n_events,5))
            fig.patch.set_facecolor('xkcd:white')

        for ev_idx,k in enumerate(event_times_dict.keys()):
            t_on = event_times_dict[k]
            r = get_binned_rasters(spikes.times,spikes.clusters,clus_ids,t_on,**self.raster_kwargs)
            stim = r.rasters[:,:,r.tscale>=0]
            t_obs = get_test_statistic(stim,blank,permute_seed=None)
            t_null = [get_test_statistic(stim,blank,permute_seed=s)[:,np.newaxis] for s in range(n_shuffles)]
            t_null = np.concatenate(t_null,axis=1)
            # implement plotting just to check
            if plotting: 
                [ax[n,ev_idx].hist(t_null[n,:],bins=int(n_shuffles/20),alpha=.7,color='k') for n in range(n_neurons)]
                [ax[n,ev_idx].axvline(t_obs[n],color = 'r') for n in range(n_neurons)]
                [off_axes(ax[n,ev_idx]) for n in range(n_neurons)]
                ax[0,ev_idx].set_title(k,rotation=45)
                if ev_idx==0:
                    [ax[n,ev_idx].text(0,0,'%.0d' % subselect_neurons[n]) for n in range(n_neurons)]


            p_values = np.array([(((t_null[n_idx,:]>t_obs_i).sum())/n_shuffles) for n_idx,t_obs_i in enumerate(t_obs)])
            p_value_per_event.append(p_values[:,np.newaxis])

        p_value_per_event = np.concatenate(p_value_per_event,axis=1)
        
        p_value_per_event = pd.DataFrame(p_value_per_event,columns=list(event_times_dict.keys()))

        if savepath:
            p_value_per_event.to_csv(savepath)

        return p_value_per_event

