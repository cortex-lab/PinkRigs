# %% 
# this script is there to look at blank trials only and look at whether on these trials what the wheen trace looks like

import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.pyutils.plotting import off_axes,off_topspines
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV038','AV036']
#subject = ['AV046']
#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2023-04-01:2023-07-28',expDef='multiSpace',data_name_dict=data_dict)
recordings = recordings[recordings.extractEvents=='1']
# throw away sessions without any laser    
to_keep = [rec.events._av_trials.is_noStimTrial.sum()>5 for _,rec in recordings.iterrows() ]
recordings = recordings[to_keep]

def sort_curr_condition(ev):
        passcond = (ev.is_blankTrial & ev.is_laserTrial & (~np.isnan(ev.timeline_choiceMoveOn)) &
                     (((ev.stim_laser1_power+ev.stim_laser2_power)==10))) #|((ev.stim_laser1_power+ev.stim_laser2_power)==17)))
        return passcond

to_keep = [((sort_curr_condition(rec.events._av_trials)).sum())>5 for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]



def batch_rasters(rec): 
    # sort the indices
    print(rec.expFolder)
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    idxs = np.where(
        sort_curr_condition(ev)
        )[0]

    powers = ev.laser_power[idxs]

    my_wheel = wheel_raster(
        ev,
        selected_trials=idxs, 
        align_type='laserOff',
        t = [-.2,0.4]
        )
    
    # reverse raster trace based on location -- would become obsolete but reverse opto does not reverse the entire trace, this is hence this exists.
    posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    r = posarg * my_wheel.rasters


    # release rt and laserstimDiff
    laserStimDiff = np.nanmin(np.array([ev.timeline_audPeriodOn,ev.timeline_visPeriodOn]),axis=0)-ev.timeline_laserOn_rampStart
        

    return r,ev.rt[idxs],powers,laserStimDiff[idxs]

r,rts,p,lsdiff = zip(*[batch_rasters(rec) for _,rec in recordings.iterrows()])
rasters = np.concatenate(r)
rts,p,lsdiff= np.concatenate(rts),np.concatenate(p),np.concatenate(lsdiff)

# %% 
_,ax = plt.subplots(1,1,figsize=(5,5))
myp=10
cr,dr = rts[p==myp],lsdiff[p==myp]
ax.plot(cr,dr,'.')
off_topspines(ax)
ax.set_xlabel('RT from stim onset (s)')

ax.set_ylabel('stim onset - LED onset (s)')
ax.set_title('Pearsons r=%.3f' % np.corrcoef(cr,dr)[0,1])
ax.set_ylim([.15,.23])
# %% 
# plot the rasters sorted by the lsdiff 


fig,ax = plt.subplots(1,1)
ax.matshow(rasters[np.argsort(lsdiff),:],aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
ax.plot(200,0,marker='v',markersize=30,color='lime')
off_axes(ax)
ax.hlines(rasters.shape[0]*0.99,350,400,'k',lw=6)

# 
# %%
fig,ax = plt.subplots(1,1)



#ax.imshow(np.abs(rasters[np.argsort(td),:]),aspect='auto',vmin=0,vmax=1)
ax.matshow(rasters[np.argsort(rasters[:,300:].mean(axis=1)),:],aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
off_axes(ax)
ax.plot(100,0,marker='v',markersize=30,color='lime')
#ax.plot(300,0,marker='v',markersize=30,color='grey')
ax.vlines(0,1,100,'k',lw=6)
ax.hlines(rasters.shape[0]*0.99,400,499,'k',lw=6)
plt.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\optoWheel_noStim.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)