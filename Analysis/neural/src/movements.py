
import sys
import numpy as np
import pandas as pd
import scipy
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat,load_data
from Analysis.neural.utils.spike_dat import bincount2D, cross_correlation
from Analysis.pyutils.batch_data import get_data_bunch


class movement_correlation():
    def __init__(self): 

        if ~hasattr(self, 'vid_dat'):    

            videos = get_data_bunch('naive-video-set')

            vid_dat = pd.DataFrame()
            for _,v in videos.iterrows():
                rec = load_data(data_name_dict={'frontCam':{'camera':['times','ROIMotionEnergy']}}, **v)
                vid_dat = pd.concat((vid_dat,rec)) 
            to_keep_calumn = [(hasattr(v.frontCam.camera,'ROIMotionEnergy') & hasattr(v.frontCam.camera,'times')) for _,v in vid_dat.iterrows()]
            vid_dat = vid_dat[to_keep_calumn]

            self.vid_dat = vid_dat

    def session_permutation(self,rec_info,t_bin):

        ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'_av_IDs'}
        other_ = {'events': {'_av_trials': 'table'},'frontCam':{'camera':['times','ROIMotionEnergy']}}

        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**rec_info)
        recordings = recordings.iloc[0]

        _,spikes,_,_,cam = simplify_recdat(recordings,probe='probe')
        clus_ids = recordings.probe.clusters._av_IDs.astype('int') 

        is_correlating = np.empty(clus_ids.size)

        r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin=t_bin)

        correlations = np.empty((len(self.vid_dat),r.shape[0]))
        is_correlated = np.zeros((clus_ids.size,1))
        actual_correlation = np.zeros((clus_ids.size,1))

        if hasattr(cam,'ROIMotionEnergy') & hasattr(cam,'times'):
            actual_idx =  np.where(self.vid_dat.expFolder == recordings.expFolder)[0][0]

            for idx,(_,v) in enumerate(self.vid_dat.iterrows()):
                cam = v.frontCam.camera
                x,y = cam.times,cam.ROIMotionEnergy
                interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
                camtrace = interp_func(t_bins[t_bins<x[-1]])
                r_ = r[:,t_bins<x[-1]]
                correlations[idx,:] = cross_correlation(camtrace,r_.T)

            actual = np.tile(correlations[actual_idx],(correlations.shape[0]-1,1))
            shuffled = correlations[~(self.vid_dat.expFolder == recordings.expFolder),:]

            is_correlating = (actual<shuffled).sum(axis=0) == 0

            for cidx,c in enumerate(clus_ids): 
                clus_ind = np.where(clus==c)[0]
                if len(clus_ind)>0:
                    actual_correlation[cidx] = actual[0,clus_ind]
                    is_correlated[cidx] = is_correlating[clus_ind]
                else: 
                    is_correlated[cidx] = False

        return actual_correlation,is_correlated