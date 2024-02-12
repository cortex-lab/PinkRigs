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
from Analysis.pyutils.plotting import off_axes,off_topspines,brainrender_scattermap
from Analysis.neural.utils.spike_dat import bombcell_sort_units,get_subregions,anatomy_plotter
from kernel_utils import load_VE_per_cluster
from Processing.pyhist.helpers.util import add_gauss_to_apdvml

dataset = 'trained-active-all'
fit_tag = 'additive-fit'
clusInfo = load_VE_per_cluster(dataset,fit_tag)
#%%
clusInfo['bc_class'] = bombcell_sort_units(clusInfo,min_spike_num=300,min_amp=20)
clusInfo['region_name'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017,mode='Beryl')
clusInfo['is_SC'] = np.array(['SC' in r for r in clusInfo.region_name])

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
clusInfo['gauss_ap'] = allen_pos_apdvml[:,0]
clusInfo['gauss_dv'] = allen_pos_apdvml[:,1]
clusInfo['gauss_ml'] = allen_pos_apdvml[:,2]


# subselectSC only clusters
gSC = clusInfo[(clusInfo.bc_class=='good') & (clusInfo.is_SC)]
# %%
from pathlib import Path
fig,ax = plt.subplots(1,1,figsize=(3,3))
fig.patch.set_facecolor('xkcd:white')


plot_examples = True
# ## naive data example neurons
example_neurons_folders = [
    '\\\\zaru.cortexlab.net\\Subjects\\AV030\\2022-12-07\\2',
    '\\\\zinu.cortexlab.net\\Subjects\\AV008\\2022-03-10\\1', 
    '\\\\zaru.cortexlab.net\\Subjects\\AV034\\2022-12-09\\1', 
]

example_neurons_probes = [
    'probe1',
    'probe1',
    'probe0',
]

example_neurons_IDs  = [
    350,
    4,
    24,
]


example_neurons_markers = [
    'd','h','X'
]

xname = 'vis'
yname = 'move_kernel_dir'

x = gSC['kernelVE_%s' % xname]
#x = gSC['kernelVE_aud']
y = gSC['kernelVE_%s' % yname]

ax.scatter(x,y,s=55,alpha=0.7,edgecolors='k',lw=1.5,c='lightgrey')

if plot_examples:
    for e,p,mid,m in zip(example_neurons_folders,example_neurons_probes,example_neurons_IDs,example_neurons_markers): 
        isc = (gSC.expFolder==e) & (gSC.probe==p) & (gSC._av_IDs==mid) 
        ax.scatter(x[isc],y[isc],s=195,alpha=1,edgecolors='k',lw=2.5,marker=m,c='r')

# replot the example neurons 
# which are: potnentially: 
#ax.set_title('r = %.2f' % np.corrcoef(x[(~np.isnan(x)) & (~np.isnan(y))],y[(~np.isnan(x)) & (~np.isnan(y))])[0,1])
isnotnan = ~np.isnan(x) & ~np.isnan(y)
print(ss.spearmanr(x[isnotnan],y[isnotnan]))
ax.set_title('Spearman r = %.2f' % ss.spearmanr(x[isnotnan],y[isnotnan]).correlation)
ax.text(.2,.2,'%.0d neurons' % gSC.shape[0])
# ax.set_xlim([-.1,.25])
# ax.set_ylim([-.05,.25])
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
off_topspines(ax)


which_figure = '%s_vs_%s'% (xname,yname)
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dataset + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)



# %%
# are there actual visual units in these recordings? If not where are they??

# %%
import plotly.express as px
px.scatter(gSC,x='kernelVE_aud_dir',y='kernelVE_vis',color='expFolder',symbol='probe',hover_data=['expFolder','probe','_av_IDs'],width=1200,height=800)
# %%
thr = 0.02

kernel_name = 'kernelVE_move_kernel'
sig = gSC[gSC[kernel_name]>thr]

anat = anatomy_plotter()
fig, ax = plt.subplots(1,1,figsize=(4,4))
fig.patch.set_facecolor('xkcd:white')




x =np.log2(sig[kernel_name])
dot_colors = brainrender_scattermap(x,vmin = np.min(x),vmax=np.max(x),n_bins=15,cmap='RdPu')


anat.plot_anat_canvas(ax=ax,axis = 'ap',coord = 3600)
anat.plot_points(gSC.gauss_ml.values, gSC.gauss_dv.values,s=20,color='grey',alpha=0.3,unilateral=True)
anat.plot_points(sig.gauss_ml.values, sig.gauss_dv.values,s=40,color=dot_colors,alpha=1,edgecolors='k',unilateral=True)

ax.set_xlim([-2200,0])
ax.set_ylim([-3100,-700])


which_figure = '%s_thr%.2f_anatomy'% (kernel_name,thr)
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dataset + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %% plot for the sake of saving out the colorbar of the same scale
fig,ax = plt.subplots(1,1)
plt.scatter(sig.gauss_ml.values, sig.gauss_dv.values,c = x,cmap='RdPu')
plt.colorbar()


which_figure = 'colorbar_for_anatomy'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dataset + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%

fig,ax = plt.subplots(1,1,figsize=(1,5))
# insert barplot for SCs and SCm

def plot_bar_ordered(regNames,desired_order = ['SCs', 'SCm'],**kwargs):
    value_counts = regNames.value_counts()
    value_counts = value_counts[desired_order]
    ax.bar(value_counts.index, value_counts.values,**kwargs)

# Adding labels and title
plot_bar_ordered(gSC.region_name,color='grey',alpha=0.5)
plot_bar_ordered(sig.region_name,color='darkmagenta')
ax.set_ylabel('# of neurons')
# get example cells 
# point to example cells

which_figure = '%s_thr%.2f_barplot'% (kernel_name,thr)
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dataset + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


# %%
sig = gSC[gSC['kernelVE_move_kernel_dir']>thr]
move_kernels = np.concatenate([np.array(r)[np.newaxis,:] for r in sig.move_kernel])
move_dir_kernels = np.concatenate([np.array(r)[np.newaxis,:] for r in sig.move_kernel_dir])

# %%
hemisphere_matrix = np.tile(sig.hemi.values[:,np.newaxis],move_dir_kernels.shape[1])
move_dir_hemi  = move_dir_kernels*hemisphere_matrix
ampidx = np.argsort(move_dir_hemi.sum(axis=1))
plt.matshow(move_dir_hemi[ampidx,:],cmap='coolwarm')

# %%
sig['choice_dir'] = move_dir_hemi[:,-1]

px.scatter(sig,x='choice_dir',y='firing_rate',hover_data=['expFolder','probe','_av_IDs','ml'])


# %%
fig,ax = plt.subplots(1,1,figsize=(8,5))

fig.patch.set_facecolor('xkcd:white')
ax.plot(move_dir_hemi[:,:].T,color='grey')
ax.axvline(30,color='k',linestyle = '--')
ax.set_xticks([0,10,20,30,40,50])
ax.set_xticklabels(np.linspace(-0.15,0.1,6).round(2))
ax.set_xlabel('time from choice onset (s)')

which_figure = 'choice_kernels'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dataset + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

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

# %%
