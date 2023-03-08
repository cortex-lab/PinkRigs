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
from Admin.csv_queryExp import load_data
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = 'AV038'

#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2023-03-08',expDef='multiSpace',data_name_dict=data_dict)
recordings = recordings[recordings.extractEvents=='1']


def batch_rasters(ev): 
    # sort the indices
    idxs = np.where(
        ev.is_laserTrial &
        ev.is_validTrial &
        ~np.isnan(ev.timeline_choiceMoveOn)
        )[0]

    my_wheel = wheel_raster(
        ev,
        selected_trials=idxs, 
        align_type='laser',
        t = [-0.1,.3]
        )
    # sort the raster
    stim_laser_timediff = (ev.timeline_visPeriodOn[idxs] 
                                - ev.timeline_laserOn_rampStart[idxs])

    return my_wheel.rasters,stim_laser_timediff

r,td = zip(*[batch_rasters(rec.events._av_trials) for _,rec in recordings.iterrows()])
rasters = np.concatenate(r)
td = np.concatenate(td)
# %%
fig,ax = plt.subplots(1,1)
ax.imshow(np.abs(rasters[np.argsort(td),:]),aspect='auto',vmin=0,vmax=5)
ax.axvline(100,color='r')
ax.set_title('%s,inactivated_side=%s,%.0fmW,align:laser,sort:audOn-laserOn' % (subject,hemisphere,selected_power))
# %%
