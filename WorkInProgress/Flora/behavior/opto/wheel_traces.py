# %%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster


from opto_utils import query_opto
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.pyutils.plotting import off_axes
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV036','AV038']
#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2023-04-01:2023-04-28',expDef='multiSpace',data_name_dict=data_dict)
recordings = recordings[recordings.extractEvents=='1']

to_keep = [rec.events._av_trials.is_noStimTrial.sum()>5 for _,rec in recordings.iterrows() ]
recordings = recordings[to_keep]
def batch_rasters(rec): 
    # sort the indices
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    idxs = np.where(
        ev.is_noStimTrial &
        ev.is_laserTrial 
        #np.isnan(ev.timeline_choiceMoveOn)
        )[0]


    my_wheel = wheel_raster(
        ev,
        selected_trials=idxs, 
        align_type='laser',
        t = [-0.1,0.4]
        )
    # sort the raster
    stim_laser_timediff = (ev.timeline_visPeriodOn[idxs] 
                                - ev.timeline_laserOn_rampStart[idxs])
    
    # reverse raster trace based on location
    posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    r = posarg * my_wheel.rasters
    return r,stim_laser_timediff

r,td = zip(*[batch_rasters(rec) for _,rec in recordings.iterrows() ])
rasters = np.concatenate(r)
td = np.concatenate(td)
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

#ax.axvline(100,color='r')
#ax.set_title('%s,inactivated_side=%s,%.0fmW,align:laser,sort:audOn-laserOn' % (subject)) 
# %%
