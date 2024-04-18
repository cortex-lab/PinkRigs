# %%
import sys
import numpy as np
import pandas as pd
import scipy
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.utils.spike_dat import bincount2D, cross_correlation
from Analysis.pyutils.loaders import call_neural_dat



# dataset_name = 'trained-passive-cureated'
# recordings = get_data_bunch(dataset_name) # or can be with queryCSV # or write a loader functin 

class movement_correlation():
    def __init__(self,**kwargs): 
        
        self.vid_dat = call_neural_dat(**kwargs)

        if 'Passive' in self.vid_dat.expDef.iloc[0]: 
            tot_vids = 'allPassive'
        else:
            tot_vids = 'active'

        print(tot_vids)

        self.unique_vids = call_neural_dat(dataset = tot_vids,
                        spikeToInclde=False,
                        camToInclude=True,
                        recompute_data_selection=True,
                        unwrap_probes= False,
                        merge_probes=False,
                        region_selection=None,
                        filter_unique_shank_positions = False)
        
        

    def session_permutation(self,rec,exclude_stim=True,tbin=0.05):

        clus_ids = rec.probe.clusters._av_IDs.astype('int') 
        spikes = rec.probe.spikes

        is_correlating = np.empty(clus_ids.size)

        # call a bunch of video data with AV experiements 
       
        unique_vids = self.unique_vids
        
        # get unique video sessions
        actual_idx =  np.where(unique_vids == rec.expFolder)[0][0]
        r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin = tbin)

        correlations = np.empty((len(unique_vids),r.shape[0]))

        for idx,(_,v) in enumerate(unique_vids.iterrows()):
            cam = v.camera
            x,y = cam.times,cam.ROIMotionEnergy
            interp_func = scipy.interpolate.interp1d(x,y,kind='linear')

            tested_period_idx = t_bins<x[-1]
            tested_period = t_bins[tested_period_idx]
            camtrace = interp_func(tested_period)
            r_ = r[:,tested_period_idx]
            
            if exclude_stim:
                ev = v.events._av_trials
                possible_onsets = np.stack((
                    ev.timeline_audPeriodOn,
                    ev.timeline_visPeriodOn, 
                    ev.block_stimOn 
                ))
                onsets_tl = np.nanmin(possible_onsets,axis=0)
                onsets_tl = onsets_tl[onsets_tl<np.max(tested_period)]
                onsets_tl = onsets_tl - 0.1 

                onsets,_,_ = bincount2D(onsets_tl,np.ones(onsets_tl.size),xbin=tbin,xlim = [np.min(tested_period), np.max(tested_period)])

                possible_offsets = np.stack((
                    ev.timeline_audPeriodOff,
                    ev.timeline_visPeriodOff,     
                    (ev.timeline_rewardOn+0.1),
                    (ev.block_stimOn+0.2)      
                ))
                offsets_tl = np.nanmax(possible_offsets,axis=0)

                offsets_tl = offsets_tl[offsets_tl<np.max(tested_period)]
                offsets,_,_ = bincount2D(offsets_tl,np.ones(offsets_tl.size),xbin=tbin,xlim = [np.min(tested_period), np.max(tested_period)])

                offsets_tl = offsets_tl + 0.7


                trial_indices = np.ravel(np.array([np.bitwise_and(tested_period >= ts[0], tested_period <= ts[-1]) 
                                for ts in zip(onsets,offsets)])) # maybe omit these 

                # filter to non-trial periods
                r_ = r_[:,~trial_indices]
                camtrace = camtrace[~trial_indices]      

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


