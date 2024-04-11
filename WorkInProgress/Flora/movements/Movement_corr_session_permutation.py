# %%
import sys
import numpy as np
import pandas as pd
import scipy
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat,load_data
from Analysis.neural.utils.spike_dat import bincount2D, cross_correlation
from Analysis.pyutils.batch_data import get_data_bunch

from loaders import load_for_movement_correlation

class movement_correlation():
    def __init__(self,**kwargs): 
        
        self.vid_dat = load_for_movement_correlation(**kwargs)

    def session_permutation(self,rec,tbin=0.05):

        clus_ids = rec.probe.clusters._av_IDs.astype('int') 
        spikes = rec.probe.spikes

        is_correlating = np.empty(clus_ids.size)

        unique_vids = self.vid_dat.drop_duplicates('expFolder')
        
        # get unique video sessions
        actual_idx =  np.where(unique_vids == rec.expFolder)[0][0]
        r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin = tbin)

        correlations = np.empty((len(unique_vids),r.shape[0]))

        for idx,(_,v) in enumerate(unique_vids.iterrows()):
            cam = v.camera
            x,y = cam.times,cam.ROIMotionEnergy
            interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
            camtrace = interp_func(t_bins[t_bins<x[-1]])
            r_ = r[:,t_bins<x[-1]]
            correlations[idx,:] = cross_correlation(camtrace,r_.T)


        shuffled = correlations[~(unique_vids.expFolder == rec.expFolder),:]
        actual = np.tile(correlations[actual_idx],(shuffled.shape[0],1))

        pvalue = (np.abs(actual)<np.abs(shuffled)).sum(axis=0)/shuffled.shape[0]

        correlations_clusID = np.empty(clus_ids.size)*np.nan
        pvalues = np.empty(clus_ids.size)*np.nan

        for idx,c in enumerate(clus_ids): 
            clus_ind = np.where(clus==c)[0]
            if len(clus_ind)>0:
                correlations_clusID[idx] = correlations[actual_idx][clus_ind[0]]
                pvalues[idx] = pvalue[clus_ind[0]]


        return correlations_clusID,pvalues
    
    def get_corrs(self,**kwargs):
        corrv,iscorr = zip(*[self.session_permutation(rec,**kwargs) 
                             for _,rec in self.vid_dat.iterrows()])
        return corrv, iscorr


