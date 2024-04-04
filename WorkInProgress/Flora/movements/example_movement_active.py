#%%
import sys
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.pyutils.plotting import off_axes,share_lim
from Analysis.pyutils.ev_dat import index_trialtype_perazimuth
from Analysis.pyutils.video_dat import get_move_raster

kwargs ={
    'subject':['AV008'],
    'expDate':'2022-03-10', 
    'expNum': '1'
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

#
rec = recordings.iloc[0]
events = rec.events._av_trials
cam = rec.camera

def plot_helper(on_times,is_sel,ax,cam = None,nrnID_idx=None,raster_kwargs=None,c='k',plot_range=True):

        if not raster_kwargs: 
            raster_kwargs  = {
                'pre_time':0.2,
                'post_time':0.3, 
                'bin_size': 0.025,
                'sortAmp':False, 'to_plot':False,
                'baseline_subtract':False
            }
            
            
        on_time = on_times[(is_sel)]
        if on_time.size>=5:
                if nrnID_idx is not None:
                    pass 
                    # dat = get_raster(on_time,spike_type = 'data',**raster_kwargs)
                    # bin_range = dat.tscale
                    # dat = dat.raster[nrnID_idx,:,:]    
                elif cam is not None:
                    dat,bin_range,_  = get_move_raster(on_time,cam.times,cam.ROIMotionEnergy,**raster_kwargs) 

                mean = dat.mean(axis=0)
                bars = dat.std(axis=0)/dat.shape[0]
                if plot_range:
                    ax.fill_between(bin_range, mean - bars, mean + bars, color=c,alpha=.4)
                else:
                    ax.plot(bin_range,mean,color=c,linestyle='solid')   


 


#
plotted_aud_azimuth =  np.array([-60,0,60])
plotted_vis_azimuth = np.array([-60,-1000,60])
plot_stim = True
plot_move = True
plot_colors = None
sep_choice = True

stim_aud_azimuth = events.stim_audAzimuth
stim_vis_azimuth = events.stim_visAzimuth

stim_aud_azimuth[np.isnan(stim_aud_azimuth)] =-1000
stim_vis_azimuth[np.isnan(stim_vis_azimuth)] =-1000

if plotted_aud_azimuth is None: 
    plotted_aud_azimuth = np.unique(stim_aud_azimuth)
if plotted_vis_azimuth is None: 
    plotted_vis_azimuth = np.unique(stim_vis_azimuth)

n_aud_pos = plotted_aud_azimuth.size
n_vis_pos = plotted_vis_azimuth.size

if plot_stim & plot_move: 
    fig,ax=plt.subplots(n_aud_pos,n_vis_pos*2,figsize=(20,10),sharex=True,sharey=True) 
    stim_plot_inds = np.arange(0,n_vis_pos*2,2)
    move_plot_inds = np.arange(0,n_vis_pos*2,2)+1
elif plot_stim and not plot_move:
    fig,ax=plt.subplots(n_aud_pos,n_vis_pos,figsize=(10,10),sharex=True,sharey=True)    
    stim_plot_inds = np.arange(0,n_vis_pos,1)  
elif plot_move and not plot_stim:
    fig,ax=plt.subplots(n_aud_pos,n_vis_pos,figsize=(10,10),sharex=True,sharey=True)    
    move_plot_inds = np.arange(0,n_vis_pos,1)        

fig.patch.set_facecolor('xkcd:white')



if plot_colors is None:
    plot_colors = ['blue','red'] 

vazi,aazi=np.meshgrid(plotted_vis_azimuth,plotted_aud_azimuth)

for i,m in enumerate(vazi):
    for j,_ in enumerate(m):
        v = vazi[i,j]
        a = aazi[i,j]
        trialtype=index_trialtype_perazimuth(a,v,'active')

        # if 'aud' in trialtype:
        #     visazimcheck =np.isnan(self.events.stim_visAzimuth)
        # elif 'blank' in trialtype:
        #     visazimcheck =np.isnan(self.events.stim_visAzimuth)
        # else:
        #     visazimcheck = (self.events.stim_visAzimuth==v)

        if sep_choice:
            n_lines = 2
        else: 
            n_lines = 1


        for mydir in range(n_lines):                  

            is_selected_trial = (events[trialtype]==1) & (stim_aud_azimuth==a)  & (stim_vis_azimuth==v) 
            
            if sep_choice:
                is_selected_trial = is_selected_trial & (events.timeline_choiceMoveDir==mydir+1)


            if plot_stim: 
                myax = ax[n_aud_pos-1-i,stim_plot_inds[j]]

                stimOnset_time = events.timeline_audPeriodOn

                # if we discard certain onsets based on how much movement happened during these trials.

                #rkw = {'t_before': 0.05,'t_after': 0.3,'sort_idx': None}
                rkw = {
                'pre_time':0.2,
                'post_time':0.3, 
                'bin_size': 0.025,
                'sortAmp':False, 'to_plot':False,
                'baseline_subtract':False
                }
            
                is_selected_trial = is_selected_trial & ~np.isnan(stimOnset_time)

                plot_helper(stimOnset_time,is_selected_trial,myax,cam=cam,
                                    c=plot_colors[mydir],raster_kwargs=rkw)
                
                if is_selected_trial.sum()>0:
                    myax.axvline(0, color ='k',alpha=0.7,linestyle='dashed')

                off_axes(myax)
                if i==0:
                    myax.set_xlabel(v)
                if j==0:
                    myax.set_ylabel(a)    
            if plot_move:
                myax = ax[n_aud_pos -1 - i,move_plot_inds[j]]
                rkw = {
                    'pre_time':0.3,
                    'post_time':0.2, 
                    'bin_size': 0.025,
                    'sortAmp':False, 'to_plot':False,
                    'baseline_subtract':False
                }
                plot_helper(events.timeline_choiceMoveOn,is_selected_trial,myax,cam=cam,
                                    c=plot_colors[mydir],raster_kwargs=rkw)

                myax.axvline(0, color ='k',alpha=0.7,linestyle='dashed')
                off_axes(myax)              
                            
            

            ax[-1,-1].hlines(-0.1,0.25,0.35,'k')


stub = '%s_%s_%s_%sMovementDuringActive.svg' % (rec.expDate, rec.expNum, rec.subject,which)

plt.savefig((Path(r'C:\Users\Flora\Pictures\PaperDraft2024') / stub),transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
