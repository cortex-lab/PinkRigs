# this code aims to produce an average psth per depth for active and passive to see the kind of sensory responses everywhere
# %%

from ibllib.atlas import AllenAtlas
# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import zscore
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_PinkRigs_to_path
from utils.data_manager import get_recorded_channel_position
from utils.spike_dat import get_binned_rasters
from utils.plotting import off_axes, off_excepty
from src.anat import call_locations_per_depth_spacing
# ONE loader from the PinkRig Pipeline
add_PinkRigs_to_path()
from Admin.csv_queryExp import load_data


probe = 'probe0'
subject = 'AV024'
depth_spacing = 60
max_depths = 5760

## call the anatomy data
anat_data  = call_locations_per_depth_spacing(subject=subject,probe=probe,max_depths=max_depths,depth_spacing=depth_spacing)


expDef = 'postActive'
align_type = 'aud'
namestring = '%s_%s_%s' % (probe,expDef,align_type)

savepath = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
savepath = savepath / ('%s/wholeProbe' % subject)
savepath.mkdir(parents=True,exist_ok=True)

data_dict = {
    'events':{'_av_trials':'table'},
    probe:{'spikes':'all'},
    ('%s_raw' % probe):{'channels':'all'},
    }

recdat = load_data(subject=subject,expDef=expDef,expDate = 'postImplant',data_name_dict = data_dict)
# from recdat drop recordings that are too short
out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration',probe,'events']]

shank_range,depth_range = zip(*[get_recorded_channel_position(rec[('%s_raw' % probe)].channels) for _,rec in recdat.iterrows()])
   
out_dat = out_dat.assign(
        shank_range = shank_range, 
        depth_range = depth_range,
        )

out_dat = out_dat.dropna() # drop recordings with spike issues 
out_dat = out_dat.drop_duplicates(subset=['shank_range','depth_range'])
shank_bins=np.arange(0,4,1)
depth_bins =np.arange(0,max_depths-depth_spacing+1,depth_spacing) # add one for even arange
depth_bin_idxs = np.arange(depth_bins.size)

# %%
# discretise spike depths
#rec = out_dat.iloc[0]

from ibllib.atlas import AllenAtlas
brain_atlas = AllenAtlas(25)

fig,ax = plt.subplots(1,shank_bins.size*2,figsize=(4,15),sharey=True)
fig.patch.set_facecolor('xkcd:white')

plot_scaling = 5 
for _,rec in out_dat.iterrows():
    spikes = rec[probe].spikes
    ev = rec.events._av_trials


    for sh in shank_bins:
        anat_plot_idx = 2*sh
        activity_plot_idx = 2*sh+1
        # plot anatomy
        region_ids_sh = anat_data[anat_data.shank==sh].region_ids.values
        r = np.array(region_ids_sh[0])+np.array(region_ids_sh[1])
        region_info = brain_atlas.regions.get(r)
        for idx in range(depth_bins.size-1):
            d = depth_bins[idx]
            binned_d = (d/depth_spacing)*plot_scaling
            ax[anat_plot_idx].fill_between([1,1.1],d,binned_d+plot_scaling,color=region_info.rgb[idx,:]/255,alpha=1)
            off_axes(ax[anat_plot_idx])

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
        [ax[activity_plot_idx].plot(zscored_resps[i,:]+plot_scaling*i,color='k',alpha=0.5) for i in range(mean_resps.shape[0])]

        # if sh>0:
        off_axes(ax[activity_plot_idx])
        # else: 
        # off_excepty(ax[sh])
        ax[activity_plot_idx].set_ylim([0,(max_depths/depth_spacing)*plot_scaling])

fig.savefig((savepath / ('%s_move.png' % namestring)),dpi=300,bbox_inches='tight')

# %%


