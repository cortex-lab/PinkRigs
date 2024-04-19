#%%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Movement_corr_session_permutation import movement_correlation
which = 'active'
m = movement_correlation(dataset=which,
                        spikeToInclde=True,
                        camToInclude=True,
                        recompute_data_selection=True,
                        unwrap_probes= True,
                        merge_probes=False,
                        region_selection=None,
                        filter_unique_shank_positions = True)


# %%
corrv,iscorr = m.get_corrs(tbin=.1)


#%%
import pandas as pd
from Admin.csv_queryExp import format_cluster_data,bombcell_sort_units,get_subregions
from Processing.pyhist.helpers.util import add_gauss_to_apdvml

clusInfo = pd.concat([format_cluster_data(rec.probe.clusters) for _,rec in m.vid_dat.iterrows()])

#%%
thr = .0001
clusInfo['is_corr'] = np.concatenate(iscorr)<thr
clusInfo['corrVal'] = np.concatenate(corrv)

bc_class = bombcell_sort_units(clusInfo)#%%
clusInfo['is_good'] = bc_class=='good'
clusInfo['Beryl'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017.values,mode='Beryl')

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
clusInfo['gauss_ap'] = allen_pos_apdvml[:,0]
clusInfo['gauss_dv'] = allen_pos_apdvml[:,1]
clusInfo['gauss_ml'] = allen_pos_apdvml[:,2]


#     

# %%
goodClus = clusInfo[clusInfo.is_good]


plot_types = ['is_corr']
regions = goodClus.Beryl.values
regions = regions[(regions!='void') & (regions!='root')]   

unique_regions, counts = np.unique(regions, return_counts=True)
thr=50
kept_regions = unique_regions[counts > thr]
indices_to_keep = np.isin(goodClus.Beryl.values, kept_regions)

goodClus = goodClus[indices_to_keep]
nGood = goodClus.shape[0]
print('%.0f good neurons' % nGood)
#%%
from Processing.pyhist.helpers.regions import BrainRegions
reg = BrainRegions()

fig,ax = plt.subplots(1,1,figsize=(2,5),sharey=True)

fraction_good = goodClus.groupby('Beryl')['is_corr'].mean()

all_ROIs = [
    'SCs','SCm','PPN','MRN','IC','CUN','PRNr',
    'VISp','VISpm','RSPv','RSPd','RSPagl',
    'POST'
]

fraction_good_ordered = pd.Series()
# add the tested areas basically
for r in all_ROIs:
    if not (r in fraction_good.index.values):
        print(r)
        fraction_good_ordered[r]=0 
    else:
        fraction_good_ordered[r] = fraction_good[r]
# Plotting
# parents = reg.acronym2acronym(fraction_good.index, mapping='Cosmos')
# orderidx = np.argsort(parents)
# Plotting
fraction_good_ordered.plot(kind='barh', color='skyblue',ax=ax)
ax.invert_yaxis()
ax.set_title('Fraction of movement neurons per region')
ax.set_ylabel('Brain Region')
ax.set_xlabel('Fraction')
ax.set_xlim([0,1])
#ax[i_t].set_xticks(rotation=45)
#ax[i_t].set_show()

from pathlib import Path
which_figure = 'movement_correlation_across_regions' + which
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
#%%
# %%

from Analysis.neural.utils.spike_dat import anatomy_plotter
from Analysis.pyutils.plotting import brainrender_scattermap


gSC = goodClus[(goodClus.Beryl=='SCs') + (goodClus.Beryl=='SCm')]

#%%

plt.hist(np.abs(gSC.corrVal[gSC.is_corr & (gSC.Beryl=='SCm')]),bins=50,alpha=.5,density=True,cumulative=True)
plt.hist(np.abs(gSC.corrVal[gSC.is_corr & (gSC.Beryl=='SCs')]),bins=50,alpha=.5,density=True,cumulative=True)
plt.hist(np.abs(gSC.corrVal[~gSC.is_corr]),bins=100,alpha=.5,density=True,cumulative=True)



#%%
fig,ax = plt.subplots(1,1,figsize=(3,1.5))
ax.hist((gSC.corrVal[~gSC.is_corr]),histtype='stepfilled',bins=30,alpha=.7,density=True,color='k')
ax.hist((gSC.corrVal[gSC.is_corr & (gSC.Beryl=='SCs')]),histtype='stepfilled',bins=30,alpha=.7,density=True,color='magenta')
ax.hist((gSC.corrVal[gSC.is_corr & (gSC.Beryl=='SCm')]),histtype='stepfilled',bins=30,alpha=.7,density=True,color='cyan')


from pathlib import Path
which_figure = 'movement_correlation_values' + which
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#%%

fig, ax = plt.subplots(2,1,figsize=(1.5,7.5),gridspec_kw={'height_ratios':[3,.8]})
fig.patch.set_facecolor('xkcd:white')
colors = ['magenta','lightblue','k']

sig = goodClus[goodClus['is_corr']]

anat = anatomy_plotter()
x =np.log2(np.abs(sig['corrVal']))
dot_colors = brainrender_scattermap(x,vmin = np.min(x),vmax=np.max(x),n_bins=35,cmap='Greys')


anat.plot_anat_canvas(ax=ax[0],axis = 'ap',coord = 3600)
anat.plot_points(gSC.gauss_ml.values, gSC.gauss_dv.values,s=15,color='grey',alpha=0.3,unilateral=True)
anat.plot_points(sig.gauss_ml.values, sig.gauss_dv.values,s=20,color=dot_colors,alpha=1,edgecolors='k',unilateral=True)

ax[0].set_xlim([-2200,0])
ax[0].set_ylim([-7100,100])
ax[0].set_title(' movement cells in SC')

anat.plot_anat_canvas(ax=ax[1],axis = 'dv',coord = 2000)
anat.plot_points(gSC.gauss_ml.values, gSC.gauss_ap.values,s=15,color='grey',alpha=0.3,unilateral=True)
anat.plot_points(sig.gauss_ml.values, sig.gauss_ap.values,s=20,color=dot_colors,alpha=1,edgecolors='k',unilateral=True)

ax[1].set_xlim([-2200,0])
ax[1].set_ylim([-4850,-2500])



which_figure = 'movement_corr_inSC' + which
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
