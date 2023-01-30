import numpy as np
import xarray as xr 
from Analysis.neural.utils.spike_dat import bincount2D

def postactive(ev):        
        blank_times = ev.block_stimOn[ev.is_blankTrial==1]

        timepoints_blanks=xr.DataArray(blank_times[:,np.newaxis],
                                dims=('trials','timeID'),
                                coords={'timeID':['ontimes']})



        trial_no = blank_times.size
        azimuths_set = np.unique(ev.stim_visAzimuth)
        azimuths_set = azimuths_set[[not np.isnan(elem) for elem in azimuths_set]]
        contrast_set = np.unique(ev.stim_visContrast)[1:]
        spl_set = np.unique(ev.stim_audAmplitude)[1:]

        # visual
        myarray=np.zeros((azimuths_set.size,contrast_set.size,trial_no,1))

        for i,myazi in enumerate(azimuths_set):
                for j,myContrast in enumerate(contrast_set):
                        ix=np.where((ev.is_visualTrial==1) & 
                                (ev.stim_visAzimuth==myazi) &
                                (ev.stim_visContrast==myContrast)
                                )[0] 
                        try: 
                                myarray[i,j,:,0]=ev.timeline_visPeriodOn[ix]
                        except ValueError: 
                                print('%.2f deg azimuth, %.2f Contrast combination does not exist' % (myazi,myContrast))



                timepoints_visual=xr.DataArray(myarray,
                                dims=('azimuths','contrast','trials','timeID'),
                                coords={'azimuths':azimuths_set,
                                        'contrast':contrast_set,
                                        'timeID':['ontimes']})

        # auditory
        myarray=np.zeros((azimuths_set.size,spl_set.size,trial_no,1))
        for i,myazi in enumerate(azimuths_set):
                for j,mySPL in enumerate(spl_set):
                        ix=np.where((ev.is_auditoryTrial==1) & 
                                (ev.stim_audAzimuth==myazi) &
                                (ev.stim_audAmplitude==mySPL)
                                )[0] 
                        myarray[i,j,:,0]=ev.timeline_audPeriodOn[ix]

                timepoints_audio=xr.DataArray(myarray,
                                dims=('azimuths','SPL','trials','timeID'),
                                coords={'azimuths':azimuths_set,
                                        'SPL':spl_set,
                                        'timeID':['ontimes']})


        #audiovisual
        myarray=np.zeros((azimuths_set.size,azimuths_set.size,contrast_set.size,spl_set.size,trial_no,2))
        for i,myaziVIS in enumerate(azimuths_set):
                for j, myaziAUD in enumerate(azimuths_set): 
                        for k,myContrast in enumerate(contrast_set): 
                                for l,mySPL in enumerate(spl_set):
                                        ix=np.where(((ev.is_coherentTrial==1) | 
                                                        (ev.is_conflictTrial==1)) & 
                                                        (ev.stim_visAzimuth ==myaziVIS) &
                                                        (ev.stim_audAzimuth==myaziAUD) &
                                                        (ev.stim_audAmplitude==mySPL) &
                                                        (ev.stim_visContrast == myContrast)
                                                        )[0]

                                        if ix.size>0: 
                                                myarray[i,j,k,l,:,0]=ev.timeline_visPeriodOn[ix[:trial_no]]
                                                myarray[i,j,k,l,:,1]=ev.timeline_audPeriodOn[ix[:trial_no]]

                
        myarray[myarray==0]=np.nan

        MSvisAzimuth = ev.stim_visAzimuth[((ev.is_coherentTrial==1) | (ev.is_conflictTrial==1))]
        MSaudAzimuth = ev.stim_audAzimuth[((ev.is_coherentTrial==1) | (ev.is_conflictTrial==1))]
        uniquecomb=np.unique(np.array([MSvisAzimuth,MSaudAzimuth]),axis=1)
        congruent_ix=np.where(uniquecomb[0,:]==uniquecomb[1,:]) 
        incongruent_ix=np.where(uniquecomb[0,:]!=uniquecomb[1,:])

        timepoints_MS=xr.DataArray(myarray,
                                dims=('visazimuths','audazimuths','contrast','SPL','trials','timeID'),
                                coords={'visazimuths':azimuths_set,
                                        'audazimuths':azimuths_set,
                                        'contrast':contrast_set,
                                        'SPL':spl_set,
                                        'timeID':['visontimes','audontimes']},
                                attrs={'congruent_azimuths': uniquecomb[0,congruent_ix], 
                                        'incongruent_vis_azimuths': uniquecomb[0,incongruent_ix],
                                        'incongruent_aud_azimuths': uniquecomb[1,incongruent_ix]})






        return timepoints_blanks,timepoints_visual,timepoints_audio,timepoints_MS

def index_trialtype_perazimuth(a,v):
    """
    function to index into dictionnary of trial types sorted by arrange_trials based on azimuth
    a: auditory azimuth
    v: visual azimuth

    returns the type of trial: blank/vis/aud/coh/conf, string    """     
            
    if (a==0) & (v==0):
        trialtype = 'is_blankTrial'
    elif (a==0) & (v!=0):
        trialtype = 'is_visualTrial'
    elif (a!=0) & (v==0):
        trialtype = 'is_auditoryTrial'
    elif (a!=0) & (v!=0) & (v==a):
        trialtype = 'is_coherentTrial'
    elif (a!=0) & (v!=0) & (v!=a):
        trialtype = 'is_conflictTrial'
    
    return trialtype

def digitise_event_onsets(ev_times,bin_range = None,**binkwargs): 
    """
    function to digitise event onsets and if called return bin indices.
    
    """
    myev_digi,t,_ = bincount2D(ev_times,np.ones(ev_times.size),**binkwargs)       
    onset_idx = np.where(myev_digi[0,:]==1)[0]

    if bin_range: 
        t_before, t_after = bin_range[0], bin_range[1]
        t_bin = binkwargs['xbin']
        bin_range = np.arange(t_before/t_bin,t_after/t_bin).astype('int')
        bin2show  = [(bin_range + idx)[np.newaxis,:] for idx in onset_idx]
        bin2show = np.concatenate(bin2show,axis=0)
    else: 
        bin2show = None

    return myev_digi,onset_idx,bin2show
