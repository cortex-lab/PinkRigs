
#%%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Processing.pyhist.helpers.regions import BrainRegions
br = BrainRegions()
from Analysis.pyutils.plotting import brainrender_scattermap

from Analysis.neural.utils.spike_dat import call_bombcell_params,bombcell_sort_units
bc_params = call_bombcell_params()

#dat_type = 'AV025AV030AV034postactive'
dat_type = 'AV025AV030AV034multiSpaceWorld_checker_training'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position
clusInfo['mlhemi'] = ((clusInfo.ml-5600)*clusInfo.hemi)+5600

clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void'
clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')
clusInfo = clusInfo.dropna(axis=1,how='all') 
# %%
# threshold varaince explained 
thr=0.01
kernel_names = [k for k in clusInfo.keys() if 'kernelVE' in k]
bool_names = []
for k in kernel_names:
    n = k.split('_')[1]
    if len(k.split('_'))==4:
        n = n+'dir'

    if len(k.split('_'))==6:
        n = n+'dir'

    n  = 'is_%s' % (n)
    bool_names.append(n)
    clusInfo[n] = clusInfo[k].values>thr
# %

clusInfo = bombcell_sort_units(clusInfo,**bc_params)
clusGood = clusInfo[clusInfo.bombcell_class=='good']
goodSC = np.array(['SC' in myloc for myloc in clusGood.brainLocationAcronyms_ccf_2017.values])
gSC = clusGood.iloc[goodSC]
# %%

move_gSC = gSC[gSC.is_movedir]
# %%
kernel_folder = csv_path.parent / r'kernel_model\stimChoice'
foldernames_per_nrn = move_gSC[['subject','expDate','expNum','probe']]

move_dir_kernels = []
for idx,c in enumerate(move_gSC._av_IDs):

    fn = '%s_%s_%s_%s' % (tuple(foldernames_per_nrn.iloc[idx]))
    k_n = kernel_folder / fn

    cIDs = np.load(k_n / 'clusIDs.npy')
    mk = np.load(k_n / 'move_kernel_dir.npy')

    matrix_idx = np.where(cIDs==c)[0][0]
    move_dir_kernels.append(mk[matrix_idx,:][np.newaxis,:])

move_dir_kernels = np.concatenate(move_dir_kernels)
# %%
from scipy.stats import zscore
move_dir_kernels_ = np.abs(move_dir_kernels)
#%%
m = move_dir_kernels_[:,:40]
sel_ax = 1 
my_min = np.tile(m.min(axis=sel_ax),(m.shape[sel_ax],1)).T
my_max = np.tile(m.max(axis=sel_ax),(m.shape[sel_ax],1)).T

m_ = (m-my_min)/(my_max-my_min)
#%%
pos_key = 'aphemi'
fig,ax = plt.subplots(1,2,figsize=(8,10),sharey=True)
pos = move_gSC[pos_key].values
pos_idx = np.argsort(pos)
ax[0].matshow(m_[pos_idx,:],aspect='auto',cmap='inferno')

ax[1].plot(pos[pos_idx],np.arange(pos.size))
# %%
import seaborn as sns

sns.histplot(data=move_gSC,x='dv',hue = 'brainLocationAcronyms_ccf_2017')
# %%
