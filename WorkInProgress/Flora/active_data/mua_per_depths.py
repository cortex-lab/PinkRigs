# %%
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_active_and_passive
from Analysis.pyutils.ev_dat import index_trialtype_perazimuth
from Analysis.neural.utils.spike_dat import bin_mua_per_depth


session = { 
    'subject':'AV034',
    'expDate': '2022-12-07',
    'probe': 'probe0'
}
sessName = '%s_%s_%s' % tuple(session.values())
dat = load_active_and_passive(session)

# _av_shankIDs should have the shank! 
spikes = dat.multiSpaceWorld.spikes
mua_bins = bin_mua_per_depth(spikes,depth_spacing = 40, depth_min = 2400, depth_max = 3250)
from Analysis.pyutils.plotting import my_rasterPSTH,off_axes
# %%
sh = 1
spikes_times = spikes.times[spikes._av_shankIDs==sh]
spikes_depths = mua_bins.depth_ids[spikes._av_shankIDs==sh]

unique_depths = np.unique(spikes_depths)

# %% 
import matplotlib.pyplot as plt 
fig,axs = plt.subplots(unique_depths.size,1,figsize = (5,15),sharey=True)
fig.patch.set_facecolor('xkcd:white')

plot_kwargs = {
    'pre_time':0.2,
    'post_time':0.8,
    'include_PSTH':True,
    'include_raster':False,
    'n_rasters':100, 
    'baseline_subtract':True
}  

for idx,d in enumerate(unique_depths):    
    ax = axs[idx]
    my_rasterPSTH(spikes_times,  # Spike times first
                    spikes_depths,  # Then cluster ids
                    [dat.multiSpaceWorld.events.timeline_audPeriodOn],
                    d,  # Identity of the cluster we plot 
                    event_colors=['red'],rasterlw=1,ax=ax,ax1=ax,**plot_kwargs)

    off_axes(ax)
# %%
