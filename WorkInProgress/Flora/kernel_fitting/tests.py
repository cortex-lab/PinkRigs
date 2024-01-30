# %%
import sys
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt 
import plotly.express as px
from pathlib import Path


my_paths = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\AV025AV030AV034multiSpaceWorld_checker_training')
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_axes,off_topspines
from Analysis.neural.utils.spike_dat import bombcell_sort_units,get_subregions

clusInfo = pd.read_csv(my_paths / 'summary_data.csv')
clusInfo['bc_class'] = bombcell_sort_units(clusInfo)
clusInfo['region_name'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017,mode='Beryl')
clusInfo['is_SC'] = np.array(['SC' in r for r in clusInfo.region_name])

# %%
gSC = clusInfo[(clusInfo.bc_class=='good') & (clusInfo.is_SC)]

# %%

fig,ax = plt.subplots(1,1,figsize=(3,3))

xname = 'vis'
yname = 'aud_kernel_spl_0.10_dir'

x = gSC['kernelVE_%s' % xname]
#x = gSC['kernelVE_aud']
y = gSC['kernelVE_%s' % yname]

ax.scatter(x,y,s=95,alpha=0.7,edgecolors='k',lw=1.5,c='lightgrey')
#ax.set_title('r = %.2f' % np.corrcoef(x[(~np.isnan(x)) & (~np.isnan(y))],y[(~np.isnan(x)) & (~np.isnan(y))])[0,1])
isnotnan = ~np.isnan(x) & ~np.isnan(y)
print(ss.spearmanr(x[isnotnan],y[isnotnan]))
ax.set_title('Spearman r = %.2f' % ss.spearmanr(x[isnotnan],y[isnotnan]).correlation)
ax.set_xlim([-.2,.3])
ax.set_ylim([-.2,.3])
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
off_topspines(ax)


# %%
px.scatter(gSC,x='kernelVE_aud_kernel_spl_0.10_dir',y='kernelVE_vis',hover_data=['expFolder','probe','_av_IDs'])


# %%
