"""
This script triggers spikes and images on aud onset, sorts them by move amp and saves 

"""
# %%
from shutil import move
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from scipy.stats import median_abs_deviation,zscore
from Analysis.neural.utils.spike_dat import get_binned_rasters
import matplotlib.pyplot as plt
from Analysis.pyutils.plotting import off_topspines

from Admin.csv_queryExp import load_data

subject = 'FT009'
expDate = '2021-01-20'
expNum= 8
cam = 'frontCam'
probe = 'probe0'

data_dict = {
            'events':{'_av_trials':['table']}, probe:{'spikes':['times','clusters']}, 
            cam:{'camera':'all','_av_motionPCs':'all'}, 'eyeCam':{'camera':'all'}
                }
recordings = load_data(subject = subject,expDate= expDate, expNum=expNum,data_name_dict=data_dict)
# 
rec_idx = 0
events = recordings.iloc[rec_idx].events['_av_trials']
camera = recordings.iloc[rec_idx][cam]['camera']
spikes = recordings.iloc[rec_idx][probe]['spikes']
# %%
from Analysis.pyutils.video_dat import get_move_raster
from Analysis.pyutils.plotting import my_rasterPSTH

# get all auditory stimulus onsets
bin_kwargs  = {
    'pre_time':.005,
    'post_time':.5, 
    'bin_size': 0.005
}
fig,ax = plt.subplots(1,1,figsize=(8,8))
fig.patch.set_facecolor('xkcd:white')


cam_values = (camera.ROIMotionEnergy)
#cam_values = (cam_values-np.median(cam_values))/median_abs_deviation(cam_values)
onset_times = events.timeline_audPeriodOn[~np.isnan(events.timeline_audPeriodOn)]
move_raster,_,sort_idx  = get_move_raster(onset_times,camera.times,cam_values,
                                sortAmp=True,to_plot=True,**bin_kwargs,baseline_subtract=False,ax=ax)

ax.set_title('movement per trial, amp sorted')

# %%

#sort_idx = np.argsort(move_raster.sum(axis=1))
fig,ax = plt.subplots(1,1,figsize=(8,8))
plot_kwargs = {
        'pethlw':2, 'rasterlw':3, 
        'erralpha':.4, 
        'n_rasters':sort_idx.size,
        'event_colors':['k'],
        'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'red',

}

bin_kwargs  = {
    'pre_time':.2,
    'post_time':.8, 
    'bin_size': 0.005
}

fig,ax = plt.subplots(1,1,figsize=(8,8))
fig.patch.set_facecolor('xkcd:white')

cam_values = (camera.ROIMotionEnergy)
#cam_values = (cam_values-np.median(cam_values))/median_abs_deviation(cam_values)
onset_times = events.timeline_audPeriodOn[~np.isnan(events.timeline_audPeriodOn)& (events.stim_audAzimuth==60)]
move_raster,_,sort_idx  = get_move_raster(onset_times,camera.times,cam_values,
                                sortAmp=True,to_plot=True,**bin_kwargs,baseline_subtract=False,ax=None)



my_rasterPSTH(spikes.times,spikes.clusters,
            [onset_times[sort_idx]],[256], include_PSTH=False,
            **bin_kwargs,**plot_kwargs, ax = ax, ax1=ax
                )

# %%
