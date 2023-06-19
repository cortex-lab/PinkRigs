# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.pyutils.plotting import brainrender_scattermap

dat_type = 'AV025AV030AV034multiSpaceWorld_checker_training'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position



# %%

from Analysis.neural.utils.spike_dat import call_bombcell_params,bombcell_sort_units
bc_params = call_bombcell_params()


clusInfo = bombcell_sort_units(clusInfo,**bc_params)
clusGood = clusInfo[clusInfo.bombcell_class=='good']

# %%
# plot based just on depth and
thr=0.02
fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(clusGood._av_xpos,clusGood.depths,'k.',alpha=0.3)
plt.plot(clusGood._av_xpos[(clusGood.kernelVE_aud>thr)],clusGood.depths[(clusGood.kernelVE_aud>thr)],'m.',alpha=0.3)

# %%

# look at the ratio of neurons per brain region

goodSC = np.array(['SC' in myloc for myloc in clusGood.brainLocationAcronyms_ccf_2017.values])


gSC = clusGood.iloc[goodSC]
# %%
# plot kernelVE against each other

import seaborn as sns 

df = gSC[['kernelVE_aud',
       'kernelVE_baseline', 'kernelVE_vis',]]
g= sns.pairplot(df)


# %%
fig,ax = plt.subplots(1,1)
ax.plot(clusGood.kernelVE_aud,clusGood.kernelVE_vis,'ko',alpha=0.3)
ax.set_ylim([-.6,.6])
ax.set_ylim([-.6,.6])

# g.axes[0,2].set_xlim((-1,1))
# g.axes[1,2].set_ylim((-1,1))
# %%
import brainrender as br
allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# Add brain regions
which_figure = 'all_nrns'
scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.07,color='sienna')
sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

sc = scene.add_brain_region("RN",alpha=0.04,color='teal')
sc = scene.add_brain_region("MRN",alpha=0.04,color='k')
sc = scene.add_brain_region("VTA",alpha=0.04,color='y')
sc = scene.add_brain_region("IC",alpha=0.04,color='y')
sc = scene.add_brain_region("VISp",alpha=0.04,color='g')

sc = scene.add_brain_region("PRNr",alpha=0.04,color='r')

scene.add(br.actors.Points(allen_pos_apdvml, colors='grey', radius=20, alpha=0.3))


thr = 0.02
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_vis>thr)], colors='blue', radius=20, alpha=1))
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_aud>thr)], colors='magenta', radius=20, alpha=1))


scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel>thr)], colors='k', radius=20, alpha=1))
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel_dir>thr)], colors='orange', radius=20, alpha=1))

scene.render()
