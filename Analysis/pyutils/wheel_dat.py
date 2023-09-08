import numpy as np
import matplotlib.pyplot as plt

from Admin.csv_queryExp import Bunch


def plot_single_trace(ev,trial_index,blfrom='first',align_time = None,ax=None,**plotkwargs):
    """
    function to plot single wheel traces with their own timings. But I think the wheel_raster func is better

    Parameters:
    -----------
    ev: Bunch
    trial_index: float 
    
    blfrom: str
        the period that we define as baseline. 
        Either before first movement or choice movement ('first'/'choice')
    
    align_time: str
        what to align the wheel times to

    """

    if 'choice' in blfrom: 
        start_move_idx = np.where(ev.timeline_wheelTime[trial_index]==ev.timeline_choiceMoveOn[trial_index])[0][0]
    elif 'first' in blfrom:
        start_move_idx = np.where(ev.timeline_wheelTime[trial_index]==ev.timeline_firstMoveOn[trial_index])[0][0]

    baseline = ev.timeline_wheelValue[trial_index][start_move_idx]

    if align_time:
        if 'aud'  in align_time:
            bl_time =  ev.timeline_audPeriodOn[trial_index]
        if 'laser' in align_time:
            bl_time = ev.timeline_laserOn_rampStart[trial_index]
    else: 
        align_time = ev.timeline_wheelTime[trial_index][start_move_idx]

    if not ax: 
        _,ax = plt.subplots(1,1)
    
    ax.plot(ev.timeline_wheelTime[trial_index]-bl_time,ev.timeline_wheelValue[trial_index]-baseline,**plotkwargs)

def wheel_raster(ev,selected_trials='all',align_type=None,t=None,t_bin=0.01, bl_subtract=True):
    """
    Function to create a rasterised version of the wheel deg per trial

    Parameters:
    ----------- 
    ev: Bunch 
        the input data requred, i.e. the event structure. 
        Could make it more generic 

    selected_trials: bool/inds
        whether a trial is inluded or not
        If passing inds, can be used for sorting
    

    """    
    if 'all' in selected_trials: 
        selected_trials = (np.ones(ev.is_validTrial.size)).astype('bool')

    if not t: 
        t=np.arange(-0.1,1,t_bin)
    else:
        t=np.arange(t[0],t[1],t_bin)

    if align_type:
        if 'aud' in align_type:
            align_time =  ev.timeline_audPeriodOn
        elif 'laserOn' in align_type:
            align_time = ev.timeline_laserOn_rampStart
        elif 'laserOff' in align_type:
            align_time = ev.timeline_laserOff_rampEnd
        elif 'vis' in align_type:
            align_time = ev.timeline_visPeriodOn
        elif 'choice' in align_type:
            align_time = ev.timeline_choiceMoveOn
        else: 
            align_time = ev.timeline_audPeriodOn        

    # hacky way of normalising the values as they are atm in the accumulating format thoughout the session
    #bl_subtracted_wheelValue = np.array([i - i[0] for i in ev.timeline_wheelValue])

    raster = np.interp(align_time[selected_trials,np.newaxis]+t,
                            np.concatenate(ev.timeline_wheelTime[selected_trials]),
                            np.concatenate(ev.timeline_wheelValue[selected_trials]))

    if bl_subtract: 
        zero_idx = np.argmin(np.abs(t))
        bl = np.nanmean(raster[:,:zero_idx],axis=1)
        bl = np.tile(bl[:,np.newaxis],t.size)
        raster = raster - bl

    return Bunch({'rasters': raster, 'tscale': t, 'cscale': selected_trials})
