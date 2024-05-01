# %%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from loaders import load_for_ccCP
recordings = load_for_ccCP(recompute_data_selection=True)


#%%
from Analysis.neural.src.cccp import cccp,get_default_set

pars = get_default_set(which='single_bin')

def run_all_ccCPs(rec,pars):
    c = cccp()
    c.load_and_format_data(rec=rec)
    c.aud_azimuths=[0]
    _,p,_,_ = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])

    return p[0],p[1],p[2],p[3]



pA,pC,pV,pB = zip(*[run_all_ccCPs(rec,pars) for _,rec in recordings.iterrows()])


# %% 
import pandas as pd
from Admin.csv_queryExp import format_cluster_data,bombcell_sort_units,get_subregions
from Processing.pyhist.helpers.util import add_gauss_to_apdvml

clusInfo = pd.concat([format_cluster_data(rec.probe.clusters) for _,rec in recordings.iterrows()])
thr = .05/4
clusInfo['is_aud'] = np.concatenate(pA)<thr
clusInfo['is_choice'] = np.concatenate(pC)<thr
clusInfo['is_vis'] = np.concatenate(pV)<thr
clusInfo['is_choice_prestim'] = np.concatenate(pB)<thr

bc_class = bombcell_sort_units(clusInfo)#%%
clusInfo['is_good'] = bc_class=='good'
clusInfo['Beryl'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017.values,mode='Beryl')

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
clusInfo['gauss_ap'] = allen_pos_apdvml[:,0]
clusInfo['gauss_dv'] = allen_pos_apdvml[:,1]
clusInfo['gauss_ml'] = allen_pos_apdvml[:,2]



# %%
goodClus = clusInfo[clusInfo.is_good]


plot_types = ['is_aud','is_vis','is_choice','is_choice_prestim']
regions = goodClus.Beryl.values
regions = regions[(regions!='void') & (regions!='root')]   

unique_regions, counts = np.unique(regions, return_counts=True)
thr=50
kept_regions = unique_regions[counts > thr]
indices_to_keep = np.isin(goodClus.Beryl.values, kept_regions)

goodClus = goodClus[indices_to_keep]
nGood = goodClus.shape[0]
print('%.0f good neurons' % nGood)

# make a plot of the distribution of nerons


all_ROIs = ['VISp','VISpm','RSPv','RSPd','RSPagl',
    'POST',
    'SCs','SCm','PPN','MRN','IC','CUN','PRNr'
    
]



fig,ax = plt.subplots(1,1,figsize=(3,5))
                      
tot_counts = goodClus.groupby('Beryl')['is_good'].sum()

# Plotting
tot_counts_ordered = pd.Series()

for r in all_ROIs:
    if not (r in tot_counts.index.values):
        print(r)
        tot_counts_ordered[r]=0 
    else:
        tot_counts_ordered[r] = tot_counts[r]

tot_counts_ordered.plot(kind='barh', color='skyblue',ax=ax)
ax.invert_yaxis()

from pathlib import Path
which_figure = 'neuron_counts'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#%%


kernel_thr = 0.02
fig,ax = plt.subplots(1,len(plot_types),figsize=(1*len(plot_types),3),sharex=True,sharey=True)
for i_t,t in enumerate(plot_types):    
    n = 'is_%s' % t
    fraction_good = goodClus.groupby('Beryl')[t].mean()

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
    fraction_good_ordered.plot(kind='barh', color='skyblue',ax=ax[i_t])
    ax[i_t].invert_yaxis()
    ax[i_t].set_title('%s' % t)
    ax[i_t].set_ylabel('Brain Region')
    ax[i_t].set_xlabel('Fraction')
    ax[i_t].set_xlim([0,.2])


from pathlib import Path
which_figure = 'ccCP_across_regions'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%

from Analysis.neural.utils.spike_dat import anatomy_plotter

gSC = goodClus[(goodClus.Beryl=='SCs') + (goodClus.Beryl=='SCm')]


fig, ax = plt.subplots(2,len(plot_types),figsize=(1.5*len(plot_types),7.5),sharex=True,gridspec_kw={'height_ratios': [3,.8] })
fig.patch.set_facecolor('xkcd:white')
colors = ['magenta','lightblue','k','orange']


for i_t,t in enumerate(plot_types):    

    sig = goodClus[goodClus[t]]

    anat = anatomy_plotter()
    #x =np.log2(sig[kernel_name])
    #dot_colors = brainrender_scattermap(x,vmin = np.m  in(x),vmax=np.max(x),n_bins=15,cmap='RdPu')


    anat.plot_anat_canvas(ax=ax[0,i_t],axis = 'ap',coord = 3600)
    anat.plot_points(goodClus.gauss_ml.values, goodClus.gauss_dv.values,s=10,color='grey',alpha=0.2,unilateral=True)
    anat.plot_points(sig.gauss_ml.values, sig.gauss_dv.values,s=25,color=colors[i_t],alpha=1,edgecolors='k',unilateral=True)

    ax[0,i_t].set_xlim([-2200,0])
    ax[0,i_t].set_ylim([-7100,-0])
    ax[0,i_t].set_title(' %s cells in SC' % t)

    anat.plot_anat_canvas(ax=ax[1,i_t],axis = 'dv',coord = 1800)
    anat.plot_points(goodClus.gauss_ml.values, goodClus.gauss_ap.values,s=10,color='grey',alpha=0.2,unilateral=True)
    anat.plot_points(sig.gauss_ml.values, sig.gauss_ap.values,s=25,color=colors[i_t],alpha=1,edgecolors='k',unilateral=True)

    ax[1,i_t].set_xlim([-2200,0])
    ax[1,i_t].set_ylim([-4850,-2500])


for i in range(3):
    for z in range(2): 
        ax[z,i+1].set_yticklabels([])

which_figure = 'ccCP_inSC'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
