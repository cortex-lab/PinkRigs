# %%

import sys
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import zscore

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Movement_corr_session_permutation import movement_correlation
from Admin.csv_queryExp import format_cluster_data,bombcell_sort_units

which = 'postactiveWithSpike'
m = movement_correlation(dataset=which,recompute_data_selection=False)

#%%
m.vid_dat.shape[0]

# %%
rec = m.vid_dat.iloc[7]
cvals,p = m.session_permutation(rec,tbin=1)
# %%
from Analysis.neural.utils.spike_dat import bincount2D,cross_correlation
import scipy

# plot the example trace
spikes = rec.probe.spikes
clusInfo = format_cluster_data(rec.probe.clusters)
bc_class = bombcell_sort_units(clusInfo)#%%
clusInfo['is_good'] = bc_class=='good'


r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin =1)
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

sigidx = np.where(p==0 & clusInfo.is_good)[0]
sig_clustersIDs = clusInfo._av_IDs[sigidx].values
sig_cluster_cvals = cvals[sigidx]
r_idx_sig_clusters = np.where(np.isin(clus, sig_clustersIDs))[0]



n_idx = r_idx_sig_clusters[np.argsort(sig_cluster_cvals)][-10]
f = 0
t=f+700
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
# %%

# %%
