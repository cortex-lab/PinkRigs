# %%
# for stats
import sys
import pandas as pd
import numpy as np
import scipy.stats as ss

# visualisations 
import seaborn as sns
import matplotlib.pyplot as plt

# my specialised functions
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_axes,off_topspines
from Analysis.neural.utils.spike_dat import bombcell_sort_units,get_subregions,anatomy_plotter
from kernel_utils import load_VE_per_cluster



dataset = 'trained-active-highSPL'
fit_tag = 'additive-fit'
clusInfo = load_VE_per_cluster(dataset,fit_tag)
#%%
clusInfo['bc_class'] = bombcell_sort_units(clusInfo,min_spike_num=300,min_amp=20)
clusInfo['region_name'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017,mode='Beryl')
clusInfo['is_SC'] = np.array(['SC' in r for r in clusInfo.region_name])

# subselectSC only clusters
gSC = clusInfo[(clusInfo.bc_class=='good') & (clusInfo.is_SC)]
# %%

fig,ax = plt.subplots(1,1,figsize=(3,3))

xname = 'move_kernel_dir'
yname = 'aud_kernel_spl_0.25_dir'

x = gSC['kernelVE_%s' % xname]
#x = gSC['kernelVE_aud']
y = gSC['kernelVE_%s' % yname]

ax.scatter(x,y,s=95,alpha=0.7,edgecolors='k',lw=1.5,c='lightgrey')
#ax.set_title('r = %.2f' % np.corrcoef(x[(~np.isnan(x)) & (~np.isnan(y))],y[(~np.isnan(x)) & (~np.isnan(y))])[0,1])
isnotnan = ~np.isnan(x) & ~np.isnan(y)
print(ss.spearmanr(x[isnotnan],y[isnotnan]))
ax.set_title('Spearman r = %.2f' % ss.spearmanr(x[isnotnan],y[isnotnan]).correlation)
ax.text(.2,.2,'%.0d neurons' % gSC.shape[0])
ax.set_xlim([-.25,.25])
ax.set_ylim([-.25,.25])
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
off_topspines(ax)


# %%
# are there actual visual units in these recordings? If not where are they??

# %%
import plotly.express as px

px.scatter(gSC,x='kernelVE_move_kernel_dir',y='kernelVE_vis',hover_data=['expFolder','probe','_av_IDs'])
# %%
thr = 0.015
sig = gSC[gSC['kernelVE_vis']>thr]

anat = anatomy_plotter()
_, ax = plt.subplots(1,1,figsize=(15,10))
anat.plot_anat_canvas(ax=ax,axis = 'ml',coord = 1000)
anat.plot_points(gSC.ap.values, gSC.dv.values,s=10,color='grey',alpha=0.1,unilateral=True)
anat.plot_points(sig.ap.values, sig.dv.values,s=10,color='red',alpha=1,unilateral=True)
# %%
sig = gSC[gSC['kernelVE_move_kernel_dir']>thr]
move_kernels = np.concatenate([np.array(r)[np.newaxis,:] for r in sig.move_kernels])
move_dir_kernels = np.concatenate([np.array(r)[np.newaxis,:] for r in sig.move_dir_kernels])

# %%
hemisphere_matrix = np.tile(sig.hemi.values[:,np.newaxis],move_dir_kernels.shape[1])
move_dir_hemi  = move_dir_kernels*hemisphere_matrix
ampidx = np.argsort(move_dir_hemi.sum(axis=1))
plt.matshow(move_dir_hemi[ampidx,:],cmap='coolwarm')
# %%
plt.plot(move_dir_hemi[ampidx,:].T)
# %%
def minmax_scaler(x):
    mmin = np.min(x)
    mmax = np.max(x)
    norm = (x-mmin)/(mmax-mmin)

    return norm

def bl_subtractor(x):
    return (x-x[0])/(x[-1]-x[0])

move_dir_hemi_ = np.apply_along_axis(bl_subtractor,1,move_dir_hemi[:,:50])
    
# %%
plt.matshow(move_dir_hemi_[ampidx,:])

# %%
plt.plot(move_dir_hemi[ampidx,:].T)
