# %% 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.neural.utils.data_manager import get_recorded_channel_position
from Analysis.pyutils.plotting import off_axes
from Analysis.neural.utils.spike_dat import get_binned_rasters
subject = 'AV025'
expDef = 'multiSpace'
probe = 'probe1'

data_dict = {
    'events':{'_av_trials':'table'},
    probe:{'spikes':'all'},
    ('%s_raw' % probe):{'channels':'all'},
    }

recdat = load_data(subject=subject,expDef=expDef,expDate = ['2022-11-09','2022-11-10','2022-11-11'],data_name_dict = data_dict)
shank_range,depth_range = zip(*[get_recorded_channel_position(rec[('%s_raw' % probe)].channels) for _,rec in recdat.iterrows()])


# from recdat drop recordings that are too short
out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration',probe,'events']]
# %%

depth_spacing = 60
min_depths = depth_range[0][0]
max_depths = depth_range[0][1]
shank_bins=np.arange(0,4,1)
depth_bins =np.arange(min_depths,max_depths-depth_spacing+1,depth_spacing) # add one for even arange
depth_bin_idxs = np.arange(depth_bins.size)

plot_scaling = 5 
fig = plt.figure(figsize=(15,5))
fig.patch.set_facecolor('xkcd:white')

subfigs = fig.subfigures(1,out_dat.shape[0])

for i,rec in out_dat.iterrows():
    ax = subfigs[i].subplots(1,4)
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
            ev.timeline_audPeriodOn[(ev.stim_audAzimuth==60) & ev.is_conflictTrial & (ev.timeline_choiceMoveDir==1)],
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
    subfigs[i].suptitle(rec.expDate)
        # else: 
        # off_excepty(ax[sh])
        #ax[sh].set_ylim([0,(max_depths/depth_spacing)*plot_scaling])


# %%
# average over sessions and look at conflcits only 

def get_confict_trial_indices(ev): 
    ev.stim_visContrast = np.round(ev.stim_visContrast,decimals=1) 
    conflict_indices = [
        [ev.is_conflictTrial & (ev.stim_audAzimuth==60) & (ev.stim_visContrast==0.1) & (ev.timeline_choiceMoveDir==1)],
        [ev.is_conflictTrial & (ev.stim_audAzimuth==60) & (ev.stim_visContrast==0.1) & (ev.timeline_choiceMoveDir==2)],
        [ev.is_conflictTrial & (ev.stim_audAzimuth==-60) & (ev.stim_visContrast==0.1) & (ev.timeline_choiceMoveDir==1)],
        [ev.is_conflictTrial & (ev.stim_audAzimuth==-60) & (ev.stim_visContrast==0.1) & (ev.timeline_choiceMoveDir==2)],
    ]
    return conflict_indices
    

# [rec,shank,depth,trialtype]
all_dat =[]
for i,rec in out_dat.iterrows():
    ax = subfigs[i].subplots(1,4)
    spikes = rec[probe].spikes
    ev = rec.events._av_trials

    inds = get_confict_trial_indices(ev)

    per_shank = []
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

        average_per_trialtype = []
        for conf_type in inds:    
            r = get_binned_rasters(
                spike_times_,
                spikes_depth_bin_idx,
                depth_bin_idxs,
                ev.timeline_audPeriodOn[conf_type],
                pre_time=0.2,
                post_time=0.4,
                baseline_subtract=True
            ) 
            mean_resps = r.rasters.mean(axis=0)
            #mean_resps[mean_resps.sum(axis=1)<.01,:] = np.NaN
            zscored_resps = zscore(mean_resps,axis=1)
            average_per_trialtype.append(zscored_resps[np.newaxis,:,:])

        per_shank.append(np.concatenate(average_per_trialtype)[np.newaxis,:,:,:])
    all_dat.append(np.concatenate(per_shank)[np.newaxis,:,:,:,:])

all_dat = np.concatenate(all_dat)

#%%
fig,ax = plt.subplots(1,4,figsize=(5,4),sharey=True)
fig.patch.set_facecolor('xkcd:white')
average = np.nanmean(all_dat,axis=0)
onerec = all_dat[2,:,:,:,:]
colors  =['r','b','g','k']
for sh in shank_bins:
        for i in range(2):
            dat_to_plot = average[sh,i,:,:]
            for d in range(average.shape[2]):
                ax[sh].plot(dat_to_plot[d,:]+plot_scaling*d,color=colors[i],alpha=0.7)
        off_axes(ax[sh])
        ax[sh].axvline(np.argmin(np.abs(r.tscale)),ymin=0,ymax=1,color='k',alpha=0.2)
        #ax[sh].set_ylim([(min_depths/depth_spacing)*plot_scaling,(max_depths/depth_spacing)*plot_scaling])

# %%
