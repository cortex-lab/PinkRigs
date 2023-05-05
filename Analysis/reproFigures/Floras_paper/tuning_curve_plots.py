# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
dat_type = 'naive-allen'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position


# %%
# %%
## TUNING CURVE PLOTS
tuning_types = ['vis','aud']
maps = {}
set_type = 'train'
for idx,t in enumerate(tuning_types):
    print(t)
    fig,ax = plt.subplots(1,1,figsize=(5,5))

    goodclus = clusInfo[clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo['is_%s_spatial' % t] & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])
]
    namekeys = [c for c in clusInfo.columns if ('%s_' % t in c) & ('_%s' % set_type in c)][:7]
    print(goodclus.shape)
    tcs = goodclus.sort_values('x0%s' % t)
    tcs = tcs[namekeys]

    tcs_norm = pd.DataFrame.div(pd.DataFrame.subtract(tcs,tcs.min(axis=1),axis='rows'),
        (tcs.max(axis=1)+tcs.min(axis=1)),axis='rows')                   


    im = ax.matshow(tcs_norm,aspect='auto',cmap='PuRd')
    #ax.set_ylim([240,0])
    off_axes(ax)
# 
    goodclus['pos_bin_idx'] = np.digitize(goodclus.aphemi,bins=np.arange(-1000,1000,250))
    unique_bins = np.unique(goodclus.pos_bin_idx)
    mean_per_pos = [np.mean(goodclus[goodclus.pos_bin_idx==b]['x0%s' % t]) for b in unique_bins]
    std_per_pos = [np.std(goodclus[goodclus.pos_bin_idx==b]['x0%s' % t]) for b in unique_bins]
    maps['%s_mean'% t] = mean_per_pos
    maps['%s_std' % t ] = std_per_pos

    #ax.vlines(-0.5,0,25,'k',lw=6)
    #fig.colorbar(im,ax=ax)
    fig.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\%s_tcs_%s.svg" % (t,set_type),transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
# IN REGISTER PLOT ####
plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':28})
print(len(maps['vis_mean']),len(maps['aud_mean']))
fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.scatter(maps['vis_mean'],maps['aud_mean'],s=100,marker='o',color='lightblue',edgecolors='k')
off_topspines(ax)
ax.plot([-120,120],[-120,120],'k--',alpha=0.3)

ax.set_xlim([-120,120])
ax.set_ylim([-120,120])
ax.set_xticks([-60,0,60])
ax.set_yticks([-60,0,60])

ax.set_xlabel('preferred visual azimuth')
ax.set_ylabel('preferred auditory azimuth')
ax.set_title('Pearson R = %.2f' % np.corrcoef(maps['vis_mean'],maps['aud_mean'])[1,0])
fig.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\map_register.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
### ENHANCEMENT INDICES #####
plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':28})
from Analysis.pyutils.plotting import off_exceptx
sc_clus = clusInfo[clusInfo.is_SC & clusInfo.is_good & clusInfo.is_aud_spatial & clusInfo.is_vis_spatial & clusInfo.is_both]
ei = sc_clus['enhancement_index_antipref,aud']
#ei = sc_clus['enhancement_index_pref']

fig,ax = plt.subplots(1,1,figsize=(3,3))
ax.hist(
    ei[~np.isnan(ei) & ~np.isinf(ei)],
    bins=np.arange(-2,2,0.4),color='thistle',alpha=0.5, edgecolor='black', linewidth=1.2
)
ax.set_xticks([-1,0,1])

off_topspines(ax)
#ax.set_xlabel(ei.name)
fig.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\%s.svg" % ei.name,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
