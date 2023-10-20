# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat
from Analysis.neural.utils.spike_dat import bincount2D,bombcell_sort_units,get_binned_rasters,cross_correlation

mname = 'AV030'
expDate = '2022-12-11'
probe = 'probe1'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
    'probe': probe
}

ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
other_ = {'events': {'_av_trials': 'table'}}
recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**session)
ev,spikes,clusters,_,_ = simplify_recdat(recordings.iloc[0],probe='probe')

bc_class = bombcell_sort_units(clusters)
kept_units = clusters._av_IDs[bc_class=='good']


# %%
kept_trials = ev.is_validTrial & (ev.response_direction!=0) & ~np.isnan(ev.timeline_choiceMoveOn) & ev.is_blankTrial
# %%
t_before = .2
t_after =.2
t_bin =.01
raster_kwargs = {
        'pre_time':t_before,
        'post_time':t_after, 
        'bin_size':t_bin,
        'smoothing':0,
        'return_fr':True,
        'baseline_subtract': True, 
}

r = get_binned_rasters(spikes.times,spikes.clusters,kept_units,ev['timeline_choiceMoveOn'][kept_trials],**raster_kwargs)
measures = r.rasters.mean(axis=2).T
behav = ev.response_direction[kept_trials]


# %%
avg = r.rasters[(behav==1),:,:].mean(axis=0)
plt.matshow(avg,aspect= 'auto',vmin=-4,vmax=4,cmap='coolwarm')
# %%

plt.plot(r.rasters[(behav==1),:,:].mean(axis=1).mean(axis=0))

plt.plot(r.rasters[(behav==2),:,:].mean(axis=1).mean(axis=0))

#%%
namestring = '{subject}_{expDate}_{expDef}_{probe}'.format(**session)
savepath = Path(r'C:\Users\Flora\Documents\Processed data\rastermap')
savepath = savepath / namestring
savepath.mkdir(parents=True,exist_ok=True)

np.save(savepath / 'spt.npy',measures)
np.save(savepath / 'behav.npy',behav)

#%%

from scipy.stats import zscore
measures = zscore(measures, axis=1)

nancells = np.isnan(measures).all(axis=1)

measures = measures[~nancells,:]

#%%
cval = cross_correlation(measures.T,behav[:,np.newaxis])[:,0]
# %%
_,ax = plt.subplots(2,1,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios': [6, 1]})
ax[0].matshow(measures[np.argsort(cval),:],aspect='auto',vmin=-4,vmax=4,cmap ='coolwarm')

ax[1].plot(behav,)
plt.show()
# %%
