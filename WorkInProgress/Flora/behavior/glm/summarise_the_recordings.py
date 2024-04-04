#%%

#%%
import sys,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from predChoice import format_av_trials,glmFit


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_probes=False,merge_probes=True,
                        region_selection=None)



# %%
from Admin.csv_queryExp import bombcell_sort_units,get_subregions
def get_stats(rec):
    clusters = rec.probe.clusters
    bc_class = bombcell_sort_units(clusters)#%%
    is_good = bc_class=='good'
    clusters = Bunch({k:clusters[k][is_good] for k in clusters.keys()})

    if 'mlapdv' not in list(clusters.keys()):
        region_names_in_framework=None
    else:
        region_names_in_framework = get_subregions(clusters.brainLocationAcronyms_ccf_2017,mode='Beryl')

    return is_good,region_names_in_framework


g,rn = zip(*[get_stats(rec) for _,rec in recordings.iterrows()])

# %%
print('%.0f good neurons' % np.concatenate(g).sum())
# %%

regions = np.concatenate([r for r in rn if r is not None])
regions = regions[(regions!='void') & (regions!='root')]   

unique_regions, counts = np.unique(regions, return_counts=True)
thr=300
kept_regions = unique_regions[counts > thr]
indices_to_keep = np.isin(regions, kept_regions)

# Filter the original array based on the indices
filtered_regions = regions[indices_to_keep]
# %%
import matplotlib.pyplot as plt

reg_to_plot = [ 'SCs', 'SCm','MRN', 'POST', 'RSPagl', 'RSPd', 'RSPv', 'VISp']
fig,ax = plt.subplots(figsize=(20,5))
for i,myr in enumerate(reg_to_plot):
    ax.bar(i,np.sum(filtered_regions==myr),color='grey')

ax.set_xticks(range(0,kept_regions.size))
ax.set_xticklabels(reg_to_plot)
ax.set_ylabel('no. of neurons')
# 
from pathlib import Path
which_figure = 'recorded_neurons_active'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name =  which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
