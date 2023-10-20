"""
loose idea: plot a longitudinal temporal window of bias
plot a bunch of things longitudinally on a rastermap because maybe then I can see longitudinal effects that are beyond each trial

"""
#  %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat
from Analysis.neural.utils.spike_dat import bincount2D,bombcell_sort_units,get_binned_rasters

mname = 'AV030'
expDate = '2022-12-11'
probe = 'probe0'
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

# %% 
bc_class = bombcell_sort_units(clusters)
kept_units = clusters._av_IDs[bc_class!='noise']

sp_kept =np.in1d(spikes.clusters,kept_units)

# %%
t_bin = 0.025
smoothing=.025
R,tscale,clusIDs = bincount2D(spikes.times[sp_kept],spikes.clusters[sp_kept],xbin=t_bin,xsmoothing=smoothing)

# bin events similarly...
# %%
feature_name_string = 'timeline_audPeriodOn'

def get_namestring_logform(ev,keep_trials,strname='timeline_audPeriodOn'): 
    currev = ev[strname][keep_trials]
    onset_type_ = np.empty(currev.size,dtype="object") 
    onset_type_[:] = strname
    return currev,onset_type_


def get_periods(onset=None,offset=None):
    """
    take two digital arrays where `signifiies the onset and he offset of the period
    and then make everything 1 in between 

    """

    onsets = np.where(onset[0])[0]
    offsets = np.where(offset[0])[0]
    stimPeriod_idx = np.concatenate([np.arange(t0,t1) for (t0,t1) in zip(onsets,offsets)])
    stimPeriod = np.zeros(events_digitised.shape[1])
    stimPeriod[stimPeriod_idx] = 1 

    return stimPeriod.astype('int')[np.newaxis,:]

keep_trials = (~np.isnan(ev.timeline_choiceMoveOn)) & ev.is_validTrial 
evtimes,evnames = zip(*[get_namestring_logform(ev,keep_trials,strname=mystr) for mystr in ['timeline_audPeriodOn','timeline_audPeriodOff']])
evtimes,evnames = np.concatenate(evtimes),np.concatenate(evnames)

# %%


events_digitised,_,event_names_ = bincount2D(evtimes,evnames,xbin=t_bin,xlim = [np.min(tscale), np.max(tscale)])


stimOn = events_digitised[event_names_=='timeline_audPeriodOn']
stimOff = events_digitised[event_names_=='timeline_audPeriodOff']
events_digitised = np.concatenate((events_digitised,get_periods(stimOn,stimOff)))
event_names_ = np.concatenate((event_names_,np.array(['stimPeriod'])))

delta_t =0.5
delta_tbin = int(delta_t/t_bin)
events_digitised = np.concatenate((events_digitised,get_periods(np.roll(stimOn,-delta_tbin),stimOn)))
event_names_ = np.concatenate((event_names_,np.array(['baselinePeriod'])))


# %% save out so that it can be loaded in another env

namestring = '{subject}_{expDate}_{expDef}_{probe}'.format(**session)
savepath = Path(r'C:\Users\Flora\Documents\Processed data\rastermap')
savepath = savepath / namestring
savepath.mkdir(parents=True,exist_ok=True)

np.save(savepath / 'spks.npy',R)
np.save(savepath / 'tscale.npy',tscale)
np.save(savepath / 'clusIDs.npy',clusIDs)
np.save(savepath / 'evs.npy',events_digitised)
np.save(savepath / 'evnames.npy',event_names_)


# %%
# fig,ax = plt.subplots(2,1,sharex=True)
# ax[0].matshow(R,aspect='auto',vmin=0,vmax=1,cmap='Greys')
# ax[1].plot(events_digitised.T)
# plt.show()
# basically we got to digitise the events as well


# do the same for trial start times or something 

# and plot the times around that.. # or something along those lines 


# or trigger around trial start


# # %%
# from rastermap import Rastermap
# from scipy.stats import zscore

# # spks is neurons by time
# # spks = zscore(R, axis=1)

# # fit rastermap
# model = Rastermap(n_PCs=200, n_clusters=100)

# # y = model.embedding # neurons x 1
# # isort = model.isort
# #%%
# # bin over neurons
# X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)

# # plot
# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111)
# ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# %%


