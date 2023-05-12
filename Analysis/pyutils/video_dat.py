import sys
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from pylab import cm
from pathlib import Path
# for the digitisation
import scipy.signal as signal
from sklearn.cluster import KMeans

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data
from Analysis.pyutils.plotting import off_axes,share_lim

def get_move_raster(on_times,camt,camv,pre_time=.1,post_time=1,bin_size=.005,sortPC1=False,sortAmp=False,baseline_subtract=True,ax=None,to_plot=False):
    """
    Function to rasterise behavioral measures (camera/wheel) that are 
    sampled at low frequency and thus need to be interpolated 

    Parameters: 
    -----------
    on_times: list 
        instances of onsets (same scale as camt)
    camt: np.ndarray 
        camera timepoints (same length as camv)
    camv: np.ndarray
        camera values 
    pre_time: float
        time prior to event 
    post_time: float 
        time taken after event for the raster 
    bin_size: float 
    sortPC1: bool
        whether to sort the raster based on how much each trial weighs on PC1, output is descending ?
    sortAmp: bool
        whether to sort the raster based on amplitude of movement after the event (specified by on_times). Output is ascending
    baseline_subtract: bool 
    to_plot: bool 
        can plot the raster output directly 
    ax: matplotlib.pyplot.axis
        can pass down on which axis to plot the raster output (in case of subplot)
    
    
    Returns:
    --------
    raster: np.ndarray
        len(on_times) x time 
    bin_range: np.ndarray
        timepoints at of the raster 
    sort_idx: 
        indices over trials by which raster was sorted 

    todo: 
    return mean/mad and bunch output

    """
    # If requested, input on_times in a sorted fashion
    on_times = on_times[:,np.newaxis]
    bin_range = np.arange(-pre_time,post_time,bin_size)
    zero_bin_idx = np.argmin(np.abs(bin_range))
    raster = np.interp((on_times+bin_range),np.ravel(camt),np.ravel(camv))

    sort_idx = None

    if baseline_subtract: 
        baseline = raster[:,zero_bin_idx][:,np.newaxis]
        baseline = np.tile(baseline,bin_range.size)
        raster = raster - baseline 

    if sortPC1:
        mu,_,_ = np.linalg.svd(raster[zero_bin_idx:,:],full_matrices=False)
        sort_idx = np.argsort(np.abs(mu[:,0]))
        raster = raster[sort_idx,:]

    if sortAmp:
        sort_idx=np.argsort(raster[:,zero_bin_idx:].mean(axis=1))
        raster = raster[sort_idx,:]

    if to_plot:
        if not ax:
            fig,ax = plt.subplots(1,1)
        movement_values = np.median(np.ravel(raster))
        print([movement_values*.65, movement_values*3])
        ax.imshow(raster,aspect='auto',cmap='Greys',norm=LogNorm(vmin=movement_values*.65, vmax=movement_values*3))
        #ax.matshow(raster,aspect='auto',cmap=cm.gray_r, norm=LogNorm(vmin=2500, vmax=12500))
        #ax.matshow(raster, cmap=cm.gray_r,aspect='auto')
        #ax.axvline(zero_bin_idx,color = 'r')
        ax.plot(zero_bin_idx,-2,marker='v',markersize=12,color='blue')
        off_axes(ax)

    return raster,bin_range,sort_idx

def digitise_motion_energy(camt,camv,plot_sample=False,min_off_time=.01,min_on_time =.01):
    """
    converts camera motion energy signal to digitised motion on/off
    process: 
        1. lowpass (<10Hz) 7th order Butterworth
        2. kmeans to determine min threshold 
        3. drop periods that don't last min_period time (s)

    Parameters: 
    -----------
    camt: numpy ndarray
    camv: numpy ndarray
    plot_sample: bool
    min_period_time: float64

    Returns: 
    -------
    (on_times,off_times,digitised) : (numpy ndarrays)


    """

    fs = 1/np.diff(camt).mean()
    fc = 10  # frequency cutoff
    w  = fc / (fs/2)
    b,a = signal.butter(7,w,'low')
    camv_filt = signal.filtfilt(b, a, camv)

    k_clus  = KMeans(n_clusters=5).fit(camv_filt[:,np.newaxis])
    # remove clusters with less than 2% points
    keep_idx = [(k_clus.labels_==clusIdx).mean()>.02 for clusIdx in np.unique(k_clus.labels_)]
    thresh = k_clus.cluster_centers_[keep_idx,0]
    thresh = np.min(thresh)#+np.ptp(thresh)*0.02 # determine the minimum thershold for crossing

    camv_thr = (camv_filt>thresh).astype('int')
    df_camv_thr = np.diff(camv_thr)
    df_camv_thr = np.insert(df_camv_thr,0,df_camv_thr[0])
    on_times = camt[df_camv_thr>0]
    off_times = camt[df_camv_thr<0]

    while on_times.size!=off_times.size: 
        # if stating in on state
        if camv_thr[0]>0: 
            on_times = np.insert(on_times,0,camt[0])
        elif camv_thr[-1]>0: 
            off_times = np.insert(off_times,0,camt[-1])
        else: 
            print('more on off differences than expected...') 
            break
    # getting rid of too short on-periods
    is_sel =(off_times-on_times)>min_off_time
    on_times, off_times = on_times[is_sel],off_times[is_sel]
    # also getting rid of too short off periods
    is_sel =(on_times[1:]-off_times[:-1])>min_on_time

    on_times, off_times = on_times[np.insert(is_sel,0,True)],off_times[np.insert(is_sel,-1,True)]

    # digitis e the signal
    digitised = np.concatenate([((camt>=on) & (camt<=off))[:,np.newaxis] for on,off in zip(on_times,off_times)],axis=1)
    digitised  = np.sum(digitised,axis=1)

    if plot_sample: 
        st = 2500
        en = 10500
        _,ax = plt.subplots(1,1,figsize=(20,5))
        plt.plot(camt[st:en],camv[st:en])
        plt.plot(camt[st:en],camv_filt[st:en])
        #plt.plot(camt[st:en],camv_thr[st:en])
        plt.plot(camt[st:en],digitised[st:en]*np.max(k_clus.cluster_centers_)*1.1)

    return (on_times,off_times,digitised)


def plot_triggerred_data(cameras=['eyeCam','frontCam','sideCam'],timings=None,**kwargs):
    if timings is None:
        timings = {
            'pre_time':.15,
            'post_time':0.45,
            'bin_size': .005
        }

    sort_by_rt = False
    sort_by_response = False

    cam_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
    cam_dict.update({'events':{'_av_trials':['table']}})
    recordings = load_data(data_name_dict=cam_dict,**kwargs)

    for _, rec in recordings.iterrows():
        
        events = rec.events._av_trials
        for cam in cameras:
            stub = '%s_%s_%s_%s_audTriggeredMovement.png' % (rec.expDate, rec.expNum, rec.subject, cam)
            try:
                camera = rec[cam]['camera']

                #  which camera values to plot 
                cam_times = camera.times
                if camera.ROIMotionEnergy.ndim==2:
                    cam_values = (camera.ROIMotionEnergy[:,0])    
                else: 
                    cam_values = (camera.ROIMotionEnergy)

                # plot by aud azimuth
                if 'is_validTrial' not in list(events.keys()):
                    events.is_validTrial = np.ones(events.is_auditoryTrial.size).astype('bool')
                    sort_by_rt=False
                is_selected  = events.is_validTrial & (events.stim_audAmplitude>0) 

                azimuths = np.unique(events.stim_audAzimuth)
                azimuths = azimuths[~np.isnan(azimuths)]
                #azimuths = np.array([-90,-60,-30,0,30,60,90])
                azi_colors = plt.cm.coolwarm(np.linspace(0,1,azimuths.size))
                fig,ax = plt.subplots(2,azimuths.size,figsize=(azimuths.size*3,5),gridspec_kw={'height_ratios':[1,3]},sharex=True)
                fig.patch.set_facecolor('xkcd:white')

                for azi,rasterax,meanax,c in zip(azimuths,ax[1,:],ax[0,:],azi_colors): 
                    is_called_trials = is_selected & (events.stim_audAzimuth==azi)# & (events.stim_audAmplitude==np.max(events.stim_audAmplitude))
                    # sort by reaction time 
                    on_times = events.timeline_audPeriodOn[is_called_trials & ~np.isnan(events.timeline_audPeriodOn)]
                    print(is_called_trials.sum())
                    if sort_by_rt: 
                        is_called_trials = is_called_trials & ~np.isnan(events.timeline_choiceMoveDir)
                        rt = events.timeline_choiceMoveOn - np.min([events.timeline_audPeriodOn,events.timeline_visPeriodOn],axis=0)
                        on_times = on_times[np.argsort(rt[is_called_trials])]
                    
                    if sort_by_response: 
                        # by default if we sort by rt 
                        on_times = on_times[np.argsort(events.response_feedback[is_called_trials])]

                    raster,br,idx = get_move_raster(
                        on_times,cam_times,cam_values,
                        sortAmp=True,baseline_subtract=False,
                        ax=rasterax,to_plot=True,**timings
                        )

                    meanax.plot(np.nanmean(raster,axis=0),color=c,lw=6)
                    rasterax.set_title(azi)
                    #rasterax.axvline(30,color='r')
                    off_axes(meanax)
                    off_axes(rasterax)
                    #rasterax.set_ylim([0,80])
                    meanax.hlines(0,br.size-0.1/timings['bin_size'],br.size,color='k')
                    if azi ==azimuths[-1]:
                        meanax.text(br.size-0.1/timings['bin_size'],np.ravel(raster).min()*.1,'0.1 s')
                    #meanax.set_title(azi)
                share_lim(ax[0,:],dim='y')
                
                
                #plt.show()
                plt.savefig((Path(rec.expFolder) / stub),transparent=False,bbox_inches = "tight",format='png',dpi=300)
            
            except:
                print('%s did not work.' % stub)


if __name__ == "__main__":  
   plot_triggerred_data(subject='FT008')