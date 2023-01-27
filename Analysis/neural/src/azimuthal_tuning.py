
import sys,itertools, re
import pandas as pd 
import numpy as np 

from Admin.csv_queryExp import load_data
from utils.data_manager import simplify_recdat
from utils.ev_dat import postactive
from utils.spike_dat import get_binned_rasters

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

        self.load(**rec_info)

    def load(self,probe = 'probe0', **rec_info):

        data_dict = {'events': {'_av_trials': 'table'},
            probe: {'spikes': ['times', 'clusters']}
        }
        recordings = load_data(data_name_dict=data_dict,**rec_info)
        events,self.spikes,_,_ = simplify_recdat(recordings.iloc[0],probe_dat_type=probe)
        _,self.vis,self.aud,_ = postactive(events)


    def get_tuning_curves(self,contrast = None, spl = None, which = 'vis', subselect_neurons = None,cv_split = 1):
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

        tuning_curves = []
        for cv in range(cv_split):
            azimuth_times_dict = {}
            if 'vis' in which:
                for (azimuth,d_power) in itertools.product(sorted(self.vis.azimuths.values),[contrast]):
                    azimuth_times_dict[('vis_%.0f' % azimuth)] = self.vis.sel(
                        azimuths=azimuth,contrast=d_power,
                        timeID='ontimes',trials=trials_idx[split_edges[cv]:split_edges[cv+1]]).values 
            
            elif 'aud' in which: 
                for (azimuth,d_power) in itertools.product(sorted(self.aud.azimuths.values),[spl]):
                    azimuth_times_dict[('aud_%.0f' % azimuth)] = self.aud.sel(
                        azimuths=azimuth,SPL=d_power,
                        timeID='ontimes',trials=trials_idx[split_edges[cv]:split_edges[cv+1]]).values

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
                


            


