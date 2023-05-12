# %%
import sys
import numpy as np
from scipy.stats import zscore
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_axes, share_lim
from Analysis.pyutils.video_dat import get_move_raster
import matplotlib.pyplot as plt

from Admin.csv_queryExp import load_data

subject = 'AV034'
expDate = '2022-12-15'
expNum= 1

cameras = ['eyeCam','frontCam','sideCam']

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
                on_times = events.timeline_audPeriodOn[is_called_trials]
                if sort_by_rt: 
                    is_called_trials = is_called_trials & ~np.isnan(events.timeline_choiceMoveDir)
                    rt = events.timeline_choiceMoveOn - np.min([events.timeline_audPeriodOn,events.timeline_visPeriodOn],axis=0)
                    on_times = on_times[np.argsort(rt[is_called_trials])]
                
                if sort_by_response: 
                    # by default if we sort by rt 
                    on_times = on_times[np.argsort(events.response_feedback[is_called_trials])]

                raster,br,idx = get_move_raster(
                    on_times,cam_times,cam_values,
                    sortAmp=False,baseline_subtract=False,
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
            
            stub = '%s_%s_%s_%s_audTriggeredMovement.png' % (rec.expDate, rec.expNum, rec.Subject, cam)
            plt.savefig((Path(rec.expFolder) / stub),transparent=False,bbox_inches = "tight",format='png',dpi=100)
        except: 
            print('error')
# # 
# #%%
# rec_idx = 0

# #
# # plot the motion PCs

# fig,ax = plt.subplots(3,3,figsize=(10,10))
# fig.patch.set_facecolor('xkcd:white')
# PC_idx = 0
# for x in range(3):
#     for y in range(3):
        
#         ax[x,y].imshow(mPC.weights[:,:,PC_idx],cmap='coolwarm')
#         off_axes(ax[x,y])
#         ax[x,y].set_title('PC %.0d' % PC_idx)
#         PC_idx += 1

# # sort by RT 
# # %%
# from matplotlib.colors import LogNorm
# from pylab import cm
# plt.matshow(raster,aspect='auto',cmap=cm.gray_r, norm=LogNorm(vmin=2500, vmax=25000))
# plt.colorbar()
# # %%
# aud_onsets = events.timeline_audPeriodOn[~np.isnan(events.timeline_audPeriodOn)]
# get_move_raster(aud_onsets,cam_times,cam_values,sortAmp=False,baseline_subtract=False,to_plot=True)

# # %%
# # focusing on outcome for a given stimulus
# # %% 
# # look at what baseline movement can predict. 
# # Options: rt, choice, go/nogo 

# is_selected  = events.is_validTrial & events.is_conflictTrial & (events.stim_audAzimuth==60)
# rt = events.timeline_choiceMoveOn - events.timeline_audPeriodOn
# rt = rt [is_selected]
# on_times = events.timeline_audPeriodOn[is_selected]
# timings = {
#     'pre_time':0,
#     'post_time':0.05,
#     'bin_size': .05}

# raster,_,_ = get_move_raster(on_times,cam_times,cam_values,sortAmp=False,baseline_subtract=False,to_plot=False,**timings)

# plt.plot(raster,rt,'.')
# plt.xlabel('movement')
# plt.ylabel('rt')
# # %%
# # choice options
# choice = events.timeline_choiceMoveDir[is_selected]
# possible_choices = [1,2]
# choice_color = ['blue','red']
# [plt.hist(raster[choice==choicedir],bins=100,color=choice_color[i],density=True,cumulative=True,histtype='step',lw=5) for i,choicedir in enumerate(possible_choices)]

# # %% go/nogo
# go_nogo = (~np.isnan(events.timeline_choiceMoveDir)).astype('int')[is_selected]
# go_colors = ['black','green']
# [plt.hist(raster[go_nogo==g],bins=100,color=go_colors[i],lw=5,density=True,cumulative=True,histtype='step') for i,g in enumerate(np.unique(go_nogo))]

# # %% movedir
# # left_choice = (events.timeline_choiceMoveDir[is_selected] == 1)
# # right_choice = (events.timeline_choiceMoveDir[is_selected] == 1)


# # %% histograms of the difference
# timings = {
#     'pre_time':.15,
#     'post_time':0.4,
#     'bin_size': .005
# }
# #  which camera values to plot 
# cam_times = camera.times
# roi_id = 0
# cam_values = zscore(camera.ROIMotionEnergy[:,roi_id])

# #_,_,cam_values = digitise_motion_energy(cam_times,cam_values)
# #cam_values = camera._av_motionPCs[:,roi_id,0]
# sort_by_rt = False
# sort_by_response = False

# # plot by aud azimuth
# if 'is_validTrial' not in list(events.keys()):
#     events.is_validTrial = np.ones(events.is_auditoryTrial.size).astype('bool')

# is_selected  = events.is_validTrial #& events.is_conflictTrial

# azimuths = np.unique(events.stim_audAzimuth)
# azimuths = azimuths[~np.isnan(azimuths)]
# azimuths = np.array([-60,60])
# azi_colors = ['blue','red']
# #fig,ax = plt.subplots(2,azimuths.size,figsize=(azimuths.size*3,5),gridspec_kw={'height_ratios':[1,3]},sharex=True)
# fig, ax  = plt.subplots(1,1)
# fig.patch.set_facecolor('xkcd:white')

# for azi,c in zip(azimuths,azi_colors): 
#     is_called_trials = is_selected & (events.stim_audAzimuth==azi)
#     #is_called_trials = is_called_trials & (events.timeline_choiceMoveDir==1)
#     # sort by reaction time 
#     on_times = events.timeline_audPeriodOn[is_called_trials]
#     if sort_by_rt: 
#         rt = events.timeline_choiceMoveOn - events.timeline_audPeriodOn
#         on_times = on_times[np.argsort(rt[is_called_trials])]
    
#     if sort_by_response: 
#         # by default if we sort by rt 
#         on_times = on_times[np.argsort(events.response_feedback[is_called_trials])]

#     raster,br,idx = get_move_raster(
#         on_times,cam_times,cam_values,
#         sortAmp=True,baseline_subtract=False,
#         ax=None,to_plot=False,**timings
#         )

#     ax.hist(raster.mean(axis=1),color=c,bins=2000,cumulative=True,histtype='step',alpha=0.5,lw=5)

# ax.set_xlabel('motionEnergy, 0-0.4s')
# ax.set_ylabel('ecdf(no. of trials)')
# #ax.set_xlim([0,0.5])
# # %%

# %%
