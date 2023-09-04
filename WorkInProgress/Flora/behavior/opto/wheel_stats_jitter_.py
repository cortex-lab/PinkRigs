# this is supposedly the clearned up code for the jitter statistics
#%%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV038','AV036']

#subject = ['AV038']
recordings = load_data(subject=subject,expDate='2022-07-11:2023-08-31',expDef='multiSpace',data_name_dict=data_dict)
# throw away sessions without any laser and reasonable trial numbers
recordings = recordings[recordings.extractEvents=='1']
to_keep = [(rec.events._av_trials.is_noStimTrial.sum()>5) & 
           (np.sum(rec.events._av_trials.is_laserTrial)>0) for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

def batch_rasters(rec,**wheelkwargs): 
    # sort the indices
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    my_wheel = wheel_raster(ev,**wheelkwargs)    
    # reverse raster trace based on location -- would become obsolete but reverse opto does not reverse the entire trace, this is hence this exists.
    #posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    #r = posarg * my_wheel.rasters
    r = my_wheel.rasters 

    return ev,r 

ev,r = zip(*[batch_rasters(rec, align_type='laserOn',t = [-.2,.6]) for _,rec in recordings.iterrows()])



# %%
# concatenation 
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
r = np.concatenate(r)

# %%
# filter 
is_selected_laserTrial = (ev.is_auditoryTrial & ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)) &
                     (((ev.stim_laser1_power+ev.stim_laser2_power)==10)) & (np.abs(ev.stim_laserPosition)==1))  


is_selected_nonlaserTrial = (ev.is_auditoryTrial & ~ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)))




# %%
fig,ax = plt.subplots(1,1)

raster = r[ev.is_noStimTrial & (ev.laser_power_signed==-17),:]
im=ax.matshow(raster,aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
ax.plot(200,0,marker='v',markersize=30,color='lime')
off_axes(ax)
#ax.hlines(raster.shape[0]*0.99,350,400,'k',lw=6)
ax.axvline(200,color='lime')
#ax.axvline(350,color='k')
cbar = fig.colorbar(im)
cbar.set_label('wheel movement (deg)')
cbar.set_ticks([.4,-.4])
cbar.set_ticklabels(['left','right'])
# %%
moveStimDiff = ev.timeline_firstMovePostLaserOn-np.nanmin(np.array([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]),axis=0)
_,ax = plt.subplots(1,1,figsize=(5,5))
ax.hist(moveStimDiff[is_selected_laserTrial],bins=100)
off_topspines(ax)
ax.set_xlabel('First movement post laser - stimulus onset (s)')
# this is supposedly the clearned up code for the jitter statistics
#%%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV038','AV036']

#subject = ['AV038']
recordings = load_data(subject=subject,expDate='2022-07-11:2023-08-31',expDef='multiSpace',data_name_dict=data_dict)
# throw away sessions without any laser and reasonable trial numbers
recordings = recordings[recordings.extractEvents=='1']
to_keep = [(rec.events._av_trials.is_noStimTrial.sum()>5) & 
           (np.sum(rec.events._av_trials.is_laserTrial)>0) for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

def batch_rasters(rec,**wheelkwargs): 
    # sort the indices
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    my_wheel = wheel_raster(ev,**wheelkwargs)    
    # reverse raster trace based on location -- would become obsolete but reverse opto does not reverse the entire trace, this is hence this exists.
    #posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    #r = posarg * my_wheel.rasters
    r = my_wheel.rasters 

    return ev,r 

ev,r = zip(*[batch_rasters(rec, align_type='laserOn',t = [-.2,.6]) for _,rec in recordings.iterrows()])



# %%
# concatenation 
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
r = np.concatenate(r)

# %%
# filter 
is_selected_laserTrial = (ev.is_auditoryTrial & ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)) &
                     (((ev.stim_laser1_power+ev.stim_laser2_power)==10)) & (np.abs(ev.stim_laserPosition)==1))  


is_selected_nonlaserTrial = (ev.is_auditoryTrial & ~ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)))




# %%
fig,ax = plt.subplots(1,1)

raster = r[ev.is_noStimTrial & (ev.laser_power_signed==-17),:]
im=ax.matshow(raster,aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
ax.plot(200,0,marker='v',markersize=30,color='lime')
off_axes(ax)
#ax.hlines(raster.shape[0]*0.99,350,400,'k',lw=6)
ax.axvline(200,color='lime')
#ax.axvline(350,color='k')
cbar = fig.colorbar(im)
cbar.set_label('wheel movement (deg)')
cbar.set_ticks([.4,-.4])
cbar.set_ticklabels(['left','right'])
# %%
moveStimDiff = ev.timeline_firstMovePostLaserOn-np.nanmin(np.array([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]),axis=0)
_,ax = plt.subplots(1,1,figsize=(5,5))
ax.hist(moveStimDiff[is_selected_laserTrial],bins=100)
off_topspines(ax)
ax.set_xlabel('First movement post laser - stimulus onset (s)')
# this is supposedly the clearned up code for the jitter statistics
#%%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV038','AV036']

#subject = ['AV038']
recordings = load_data(subject=subject,expDate='2022-07-11:2023-08-31',expDef='multiSpace',data_name_dict=data_dict)
# throw away sessions without any laser and reasonable trial numbers
recordings = recordings[recordings.extractEvents=='1']
to_keep = [(rec.events._av_trials.is_noStimTrial.sum()>5) & 
           (np.sum(rec.events._av_trials.is_laserTrial)>0) for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

def batch_rasters(rec,**wheelkwargs): 
    # sort the indices
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    my_wheel = wheel_raster(ev,**wheelkwargs)    
    # reverse raster trace based on location -- would become obsolete but reverse opto does not reverse the entire trace, this is hence this exists.
    #posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    #r = posarg * my_wheel.rasters
    r = my_wheel.rasters 

    return ev,r 

ev,r = zip(*[batch_rasters(rec, align_type='laserOn',t = [-.2,.6]) for _,rec in recordings.iterrows()])



# %%
# concatenation 
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
r = np.concatenate(r)

# %%
# filter 
is_selected_laserTrial = (ev.is_auditoryTrial & ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)) &
                     (((ev.stim_laser1_power+ev.stim_laser2_power)==17)) & (np.abs(ev.stim_laserPosition)==1))  


is_selected_nonlaserTrial = (ev.is_auditoryTrial & ~ev.is_laserTrial & ev.is_validTrial & (ev.stim_audAzimuth==60) & (~np.isnan(ev.timeline_choiceMoveOn)))




# %%

# %%
moveStimDiff = ev.timeline_firstMovePostLaserOn-np.nanmin(np.array([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]),axis=0)
_,ax = plt.subplots(1,1,figsize=(5,5))
ax.hist(moveStimDiff[is_selected_laserTrial],bins=100)
off_topspines(ax)
ax.set_xlabel('First movement post laser - stimulus onset (s)')
ax.set_ylabel('#trials')


# %%
moveLaserDiff = ev.timeline_firstMovePostLaserOn-ev.timeline_laserOn_rampStart

fig,ax = plt.subplots(1,1)

i = ev.is_noStimTrial & (ev.laser_power_signed==-17)
cr =r[i,:]
ax.plot((cr).T,color='blue',alpha =.05)
ax.plot(np.nanmean((cr),axis=0),color='blue',alpha =1,lw=6)



i = ev.is_noStimTrial & (ev.laser_power_signed==17)
cr =r[i,:]
ax.plot((cr).T,color='red',alpha =.05)
ax.plot(np.nanmean((cr),axis=0),color='red',alpha =1,lw=6)

ax.axvline(200,color='lime')
off_axes(ax)

# %%
fig,ax = plt.subplots(1,1)

i = (ev.is_validTrial & ev.is_laserTrial & ~ev.is_noStimTrial) & (np.abs(ev.stim_laserPosition)==1) & (np.isnan(ev.timeline_choiceMoveOn))
cr =r[i,:]
ax.plot((cr).T,color='blue',alpha =.05)
ax.plot(np.nanmean((cr),axis=0),color='blue',alpha =1,lw=6)

# %%
# %%
fig,ax = plt.subplots(1,1)

ax.hist(moveLaserDiff[(ev.is_noStimTrial)],bins=200)
ax.set_xlabel('First movement post laser - laser onset (s)')
off_topspines(ax)
# %%


fig,ax = plt.subplots(1,1)

raster = r[ev.is_noStimTrial,:]

im=ax.matshow(raster[np.argsort(moveLaserDiff[ev.is_noStimTrial])],aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
ax.plot(200,0,marker='v',markersize=30,color='lime')
off_axes(ax)
#ax.hlines(raster.shape[0]*0.99,350,400,'k',lw=6)
ax.axvline(200,color='lime')
#ax.axvline(350,color='k')
cbar = fig.colorbar(im)
cbar.set_label('wheel movement (deg)')
cbar.set_ticks([.4,-.4])
cbar.set_ticklabels(['left','right'])
# %%
