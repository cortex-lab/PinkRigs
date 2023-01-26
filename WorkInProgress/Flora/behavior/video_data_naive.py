# %%

import sys
import numpy as np
from scipy.stats import zscore
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_github_paths
from utils.plotting import off_axes, share_lim
from utils.video_dat import get_move_raster
import matplotlib.pyplot as plt


add_github_paths()

from Analysis.helpers.queryExp import load_data

subject = 'FT009'
expDate = '2021-01-20'
expNum= 8

cam = 'frontCam'
data_dict = {
            'events':{'_av_trials':['table']},
            cam:{'camera':'all','_av_motionPCs':'all'}, 
            'eyeCam':{'camera':'all'}
                }
#recordings = load_data(subject = subject,expDate= expDate, expNum=expNum,data_name_dict=data_dict)
recordings = load_data(subject = subject,expDate= expDate,expNum=expNum, data_name_dict=data_dict)

rec_idx = 0
events = recordings.iloc[rec_idx].events['_av_trials']
camera = recordings.iloc[rec_idx][cam]['camera']
mPC = recordings.iloc[rec_idx][cam]['_av_motionPCs']

# add the pupil if its available (otherwise probably just comment out)
pupil = recordings.iloc[rec_idx]['eyeCam']['camera']

#
# plot the motion PCs

fig,ax = plt.subplots(3,3,figsize=(10,10))
fig.patch.set_facecolor('xkcd:white')
PC_idx = 0
for x in range(3):
    for y in range(3):
        
        ax[x,y].imshow(mPC.weights[:,:,PC_idx],cmap='coolwarm')
        off_axes(ax[x,y])
        ax[x,y].set_title('PC %.0d' % PC_idx)
        PC_idx += 1

cam_times = camera.times
cam_values = camera.ROIMotionEnergy
aud_onsets = events.timeline_audPeriodOn[~np.isnan(events.timeline_audPeriodOn)]

raster,bin_range,sort_idx = get_move_raster(
    aud_onsets,cam_times,cam_values,
    sortAmp=True,baseline_subtract=True,to_plot=True
    )


# %%
# plot the pupil and the movement side by side.
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from pylab import cm



# get the pupil too 
pupil_raster,_,_ = get_move_raster(
    aud_onsets,pupil.times,pupil.pupil_area_smooth,
    sortAmp=False,baseline_subtract=False,to_plot=True
    )

fig,ax = plt.subplots(1,2,sharey=True,figsize=(10,10))
fig.patch.set_facecolor('xkcd:white')
#ax.matshow(raster,vmin=0,vmax=np.min(camv)+2.5,aspect='auto',cmap=cm.gray_r)
ax[0].matshow(raster, cmap=cm.gray_r, norm=LogNorm(vmin=1000, vmax=5000),aspect='auto')
ax[1].matshow(pupil_raster[sort_idx,:], cmap=cm.gray_r, norm=LogNorm(vmin=20, vmax=2000),aspect='auto')
off_axes(ax[0])
off_axes(ax[1])
# %%
