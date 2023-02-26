
import itertools, re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from Admin.csv_queryExp import load_ephys_independent_probes,Bunch
from Analysis.neural.utils.ev_dat import postactive
from Analysis.neural.utils.spike_dat import get_binned_rasters

# load data

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
        _,self.vis,self.aud,_ = postactive(events)

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
            :pd.DataFrame        

        """
        if not contrast: 
            contrast = np.max(self.vis.contrast.values)

        if not spl:
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
                           
        azimuth_times_dict = {}
        for azimuth in azimuth_options:
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
    


    def get_tuning_curves(self,rasters = None,cv_split=2,azimuth_shuffle_seed=None):
        """
        Parameters: 
        -----------
        rasters: np.ndarray 
            azimuths x trials x neurons x timebins 
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
            for k in azimuth_inds:
                curr_idxs = trials_idx[split_edges[cv]:split_edges[cv+1]]
                stim = rasters[k,curr_idxs,:,:]                 
                Rmax_stim = np.max(np.abs(stim.mean(axis=0)),axis=1)
                responses[azimuths[k]] = Rmax_stim

            curr_tuning_curves = pd.DataFrame.from_dict(responses)            
            preferred_tuning_idx = np.argmax(curr_tuning_curves.to_numpy(),axis=1)

            curr_tuning_curves[('preferred_tuning')] = [re.split('_',curr_tuning_curves.columns[i])[-1] for i in preferred_tuning_idx]
            curr_tuning_curves['cv_number'] = cv
            curr_tuning_curves = curr_tuning_curves.set_index(self.clus_ids,drop=True)
            tuning_curves.append(curr_tuning_curves)
        
        tuning_curves = pd.concat(tuning_curves)

        return tuning_curves
    

    def get_selectivity(self,**kwargs):
        """
        function to get cross-validated selectivity from responses at various azimuthal tunings
        Procedure: 

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

        return selectivity,tuning_curves[(tuning_curves.cv_number==0)].preferred_tuning.values
    
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

        return is_selective,self.preferred_tuning
            
    def plot_selectivity_distribution(self,clusID):
        cidx = np.where(self.clus_ids==clusID)[0][0]
        _,ax = plt.subplots(1,1)
        ax.hist(self.selectivity[:,cidx])
        ax.axvline(self.selectivity_shuffle_dist[cidx])
        ax.set_title(self.preferred_tuning[cidx])





            


