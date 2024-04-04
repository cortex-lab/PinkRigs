# %%
import sys
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.pyutils.plotting import off_axes,share_lim
from Analysis.pyutils.video_dat import get_move_raster

kwargs ={
    'subject':['FT031'],
    'expDate':'2021-12-03', 
    'expNum': '2'
}

timings = {
    'pre_time':.15,
    'post_time':0.45,
    'bin_size': .005
}

sort_by_rt = False
sort_by_response = False

which = 'vis'

cameras = ['frontCam','sideCam','eyeCam']
cam_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
cam_dict.update({'events':{'_av_trials':['table']}})
recordings = load_data(data_name_dict=cam_dict,**kwargs,cam_hierarchy=cameras)





rec = recordings.iloc[0]    
events = rec.events._av_trials
stub = '%s_%s_%s_%sTriggeredMovement.svg' % (rec.expDate, rec.expNum, rec.subject,which)
camera = rec['camera']


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

if which=='aud':    
    is_selected  = events.is_validTrial & (events.stim_audAmplitude==np.max(events.stim_audAmplitude)) & events.is_auditoryTrial
elif which=='vis':
    is_selected  = events.is_validTrial & (events.stim_visContrast==np.max(events.stim_visContrast)) & events.is_visualTrial

azimuths = np.unique(events.stim_audAzimuth)
azimuths = azimuths[~np.isnan(azimuths)]
#azimuths = np.array([-90,-60,-30,0,30,60,90])
azi_colors = plt.cm.coolwarm(np.linspace(0,1,azimuths.size))
fig,ax = plt.subplots(2,azimuths.size,figsize=(azimuths.size*3,5),gridspec_kw={'height_ratios':[1,3]},sharex=True)
fig.patch.set_facecolor('xkcd:white')

for azi,rasterax,meanax,c in zip(azimuths,ax[1,:],ax[0,:],azi_colors): 

    if which=='aud':    
        is_called_trials = is_selected & (events.stim_audAzimuth==azi)# & (events.stim_audAmplitude==np.max(events.stim_audAmplitude))

        on_times = events.timeline_audPeriodOn[is_called_trials & ~np.isnan(events.timeline_audPeriodOn)]

    elif which=='vis':
        is_called_trials = is_selected & (events.stim_visAzimuth==azi)# & (events.stim_audAmplitude==np.max(events.stim_audAmplitude))
        on_times = events.timeline_visPeriodOn[is_called_trials & ~np.isnan(events.timeline_visPeriodOn)]

    # sort by reaction time 


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
    
    ax[1,0].vlines(0,100,0,'k') # mark 10 trials

    meanax.plot(np.nanmean(raster,axis=0),color=c,lw=6)
    rasterax.set_title(azi)
    #rasterax.axvline(30,color='r')
    off_axes(meanax)
    #off_axes(rasterax)
    #rasterax.set_ylim([0,80])
    meanax.hlines(0,br.size-0.1/timings['bin_size'],br.size,color='k')
    if azi ==azimuths[-1]:
        meanax.text(br.size-0.1/timings['bin_size'],np.ravel(raster).min()*.1,'0.1 s')
    #meanax.set_title(azi)
share_lim(ax[0,:],dim='y')

            
            #plt.show()
plt.savefig((Path(r'C:\Users\Flora\Pictures\PaperDraft2024') / stub),transparent=False,bbox_inches = "tight",format='svg',dpi=300)
        
# %%
