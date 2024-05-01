# %%

import sys
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import zscore

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Movement_corr_session_permutation import movement_correlation
from Admin.csv_queryExp import format_cluster_data,bombcell_sort_units

which = 'naive'
m = movement_correlation(dataset=which,
                        spikeToInclde=True,
                        camToInclude=True,
                        recompute_data_selection=True,
                        unwrap_probes= True,
                        merge_probes=False,
                        region_selection=None,
                        filter_unique_shank_positions = True)

#%%
m.vid_dat.shape[0]

# %%
T_BIN =.1
rec = m.vid_dat.iloc[2]
cvals,p = m.session_permutation(rec,tbin=T_BIN)
# %%
from Analysis.neural.utils.spike_dat import bincount2D,cross_correlation
import scipy

# plot the example trace
spikes = rec.probe.spikes
clusInfo = format_cluster_data(rec.probe.clusters)
bc_class = bombcell_sort_units(clusInfo)#%%
clusInfo['is_good'] = bc_class=='good'


r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin =T_BIN)
#
cam = rec.camera
x,y = cam.times,cam.ROIMotionEnergy
interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
times = t_bins[t_bins<x[-1]] # cut to the end the camera recording 
camtrace = interp_func(times)
r_ = zscore(r[:,t_bins<x[-1]],axis=1)

cvals_check = cross_correlation(camtrace,r_.T)

# %%

plt.hist(cvals[p==0],bins=70,alpha=.5)
plt.hist(cvals[p>0],bins=70,alpha=.5)

# %%

# plot the actual the trace

fig,ax = plt.subplots(2,1,figsize=(20,2),sharex=True)

sigidx = np.where((p==0) & clusInfo.is_good & 
                  ((clusInfo.BerylAcronym=='SCs') + (clusInfo.BerylAcronym=='SCm')))[0]

sig_clustersIDs = clusInfo._av_IDs[sigidx].values
sig_cluster_cvals = cvals[sigidx]
r_idx_sig_clusters = np.where(np.isin(clus, sig_clustersIDs))[0]



n_idx = r_idx_sig_clusters[np.argsort(sig_cluster_cvals)][-1]
f = 9000
t=f+2000
ax[0].plot(times[f:t],camtrace[f:t])
ax[1].plot(times[f:t],r_[n_idx,f:t])
ax[0].set_title('r = %.2f' % cvals[n_idx])

# 
fig,ax = plt.subplots(2,2,figsize=(20,12), gridspec_kw={'height_ratios': [1, 5],'width_ratios':[1,7]})
ax[0,1].plot(camtrace[f:t])
ax[1,0].plot(np.sort(sig_cluster_cvals),np.arange(sig_cluster_cvals.size))
ax[1,1].matshow(r_[r_idx_sig_clusters[np.argsort(sig_cluster_cvals)],f:t],
                aspect='auto',vmin=-5,vmax=5,cmap='coolwarm')
ax[1,1].get_shared_x_axes().join(ax[1,1], ax[0,1])
ax[1,1].get_shared_y_axes().join(ax[1,1], ax[1,0])


from Analysis.pyutils.plotting import off_axes,off_topspines

off_axes(ax[0,0])
off_axes(ax[1,1])
off_topspines(ax[0,1])
off_topspines(ax[1,0])

ax[1,0].axvline(0,color='k',linestyle='--')

ax[1,1].hlines(-1,1750,1800,color='k')
print('tbar is %.2f s ' % ((np.diff(t_bins)).mean()*50))

from pathlib import Path

which_figure = 'movement_correlations_example'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = which + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.colorbar(cax=ax[0,0], ax=ax[1,1])
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Plot the data with the coolwarm colormap
plt.contourf(X, Y, Z, cmap='coolwarm')

# Add a colorbar
plt.colorbar()

# Show the plot
plt.show()
# %%
