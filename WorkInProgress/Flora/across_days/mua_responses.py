# %% 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.neural.utils.data_manager import get_recorded_channel_position
from Analysis.neural.utils.plotting import off_axes
from Analysis.neural.utils.spike_dat import get_binned_rasters
subject = 'AV025'
expDef = 'multiSpace'
probe = 'probe0'

data_dict = {
    'events':{'_av_trials':'table'},
    probe:{'spikes':'all'},
    ('%s_raw' % probe):{'channels':'all'},
    }

recdat = load_data(subject=subject,expDef=expDef,expDate = ['2022-11-11','2022-11-09'],data_name_dict = data_dict)
shank_range,depth_range = zip(*[get_recorded_channel_position(rec[('%s_raw' % probe)].channels) for _,rec in recdat.iterrows()])


# from recdat drop recordings that are too short
out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration',probe,'events']]
# %%

depth_spacing = 60
max_depths = 5760
shank_bins=np.arange(0,4,1)
depth_bins =np.arange(0,max_depths-depth_spacing+1,depth_spacing) # add one for even arange
depth_bin_idxs = np.arange(depth_bins.size)

plot_scaling = 5 

fig,ax = plt.subplots(1,4,figsize=(5,5))

for _,rec in out_dat.iterrows():
    spikes = rec[probe].spikes
    ev = rec.events._av_trials


    for sh in shank_bins:
        for idx in range(depth_bins.size-1):
            d = depth_bins[idx]
            binned_d = (d/depth_spacing)*plot_scaling


        is_good_idx = (spikes._av_shankIDs==sh)
        spike_times_ = spikes.times[is_good_idx]
        spike_depths_ = spikes.depths[is_good_idx]

        spikes_depth_bin_idx = np.digitize(
            spike_depths_,
            bins = depth_bins,
            )

        r = get_binned_rasters(
            spike_times_,
            spikes_depth_bin_idx,
            depth_bin_idxs,
            ev.timeline_audPeriodOn[ev.is_coherentTrial],
            pre_time=0.2,
            post_time=1,
            baseline_subtract=True
        )

        mean_resps = r.rasters.mean(axis=0)
        mean_resps[mean_resps.sum(axis=1)<10,:] = np.NaN
        zscored_resps = zscore(mean_resps,axis=1)
        [ax[sh].plot(zscored_resps[i,:]+plot_scaling*i,color='k',alpha=0.5) for i in range(mean_resps.shape[0])]

        # if sh>0:
        off_axes(ax[sh])
        # else: 
        # off_excepty(ax[sh])
        ax[sh].set_ylim([0,(max_depths/depth_spacing)*plot_scaling])


# %%
