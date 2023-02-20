
import sys,itertools, re
import pandas as pd 
import numpy as np 
import xarray as xr

from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat
from Analysis.neural.utils.ev_dat import postactive
from Analysis.neural.utils.spike_dat import get_binned_rasters

# load data

def get_test_statistic(tuning_curves,tuning_type = None):
    """
    utility function to get test statistic from the tuning curve table outputted by the class 
    this is is specific for the permuation test I am doing and thus requires the tuning 
    curve table to be called as above 

    """ 
    if not tuning_type:
        tuning_type = tuning_curves.columns[0][:3]
                   
    train = tuning_curves[(tuning_curves.cv_number == 0)]
    train = train[('preferred_%s_tuning' % tuning_type)]
    test = tuning_curves[(tuning_curves.cv_number == 1)]
    test = test[('preferred_%s_tuning' % tuning_type)]

    statistic = np.abs(train.values.astype('float')-test.values.astype('float'))

    return statistic


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

        self.load(**rec_info)

    def load(self, **rec_info):

        ephys_dict =  {'spikes': ['times', 'clusters']}
        other_ = {'events': {'_av_trials': 'table'}}

        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**rec_info)
        if recordings.shape[0] == 1:            
            recordings =  recordings.iloc[0]
        else:
            print('recordings are ambiguously defined. Please recall.')
        events,self.spikes = recordings.events._av_trials,recordings.probe.spikes
        _,self.vis,self.aud,_ = postactive(events)


    def get_tuning_curves(self,contrast = None, spl = None, which = 'vis', subselect_neurons = None,cv_split = 1,azimuth_shuffle=None):
        """
        get tuning curves for the loaded data

        Parameters: 
        ----------
        contrast: numpy.ndarray
        spl: numpy.ndarray
        which: str
            'vis' or 'aud'
        subselect_neurons: list
            cluster_ids
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
            clus_ids = np.unique(self.spikes.clusters)    

        # create indices for cross-validation
        trials_idx = self.aud.trials.values
        np.random.seed(0)
        trials_idx = np.random.permutation(trials_idx)
        split_edges = list(itertools.accumulate(itertools.repeat(int(trials_idx.size/cv_split),cv_split)))
        split_edges.insert(0,0)

        if 'vis' in which:
            azimuth_options = sorted(self.vis.azimuths.values)
            onset_matrix = self.vis.sel(contrast=contrast,timeID='ontimes')
        elif 'aud' in which:
            azimuth_options = sorted(self.aud.azimuths.values)
            onset_matrix = self.aud.sel(SPL=spl,timeID='ontimes')      

        if azimuth_shuffle:
            # shuffle the onset matrix. (eqivalent to shuffling azimuth labels)   
            np.random.seed(azimuth_shuffle)  # seed ID of the azimuthal shuffing
            permuted_onset_index = np.random.permutation(np.ravel(onset_matrix))
            onset_matrix = xr.DataArray(permuted_onset_index.reshape(onset_matrix.shape),
                                dims=('azimuths','trials'),
                                coords={'azimuths':azimuth_options})

            

        tuning_curves = []
        for cv in range(cv_split):
            azimuth_times_dict = {}
            for azimuth in azimuth_options:
                azimuth_times_dict[('%s_%.0f' % (which,azimuth))] = onset_matrix.sel(
                    azimuths = azimuth,
                    trials=trials_idx[split_edges[cv]:split_edges[cv+1]]
                ).values

            responses = {}
            for k in azimuth_times_dict.keys():
                t_on = azimuth_times_dict[k]
                r = get_binned_rasters(self.spikes.times,self.spikes.clusters,clus_ids,t_on,**self.raster_kwargs)
                stim = r.rasters[:,:,r.tscale>=0]
                Rmax_stim = np.max(np.abs(stim.mean(axis=0)),axis=1)
                responses[k] = Rmax_stim

            curr_tuning_curves = pd.DataFrame.from_dict(responses)            
            preferred_tuning_idx = np.argmax(curr_tuning_curves.to_numpy(),axis=1)

            curr_tuning_curves[('preferred_%s_tuning' % which)] = [re.split('_',curr_tuning_curves.columns[i])[-1] for i in preferred_tuning_idx]
            curr_tuning_curves['cv_number'] = cv
            curr_tuning_curves = curr_tuning_curves.set_index(clus_ids,drop=True)


            tuning_curves.append(curr_tuning_curves)

        tuning_curves = pd.concat(tuning_curves)

        return tuning_curves





            


