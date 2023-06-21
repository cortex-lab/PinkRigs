# some exploratory analsis looking at sensitivity/disriminability for the same neurons in active vs passive
# probably should calculate only for neurons that pass quality metrics, so
# calulate selectivity index in active vs passive for each neuron 
# %%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
from Admin.csv_queryExp import load_ephys_independent_probes,Bunch,simplify_recdat
from Analysis.neural.utils.spike_dat import get_binned_rasters


# use easiser set of code, i.e. just load the events and look and L-R for all neurons at 60 deg  

mname = 'AV030'
expDate = '2022-12-09'
probe = 'probe1'

raster_kwargs = {
                'pre_time':0.2,
                'post_time':0.50, 
                'bin_size':0.01,
                'smoothing':0.025,
                'return_fr':True,
                'baseline_subtract': False, 
        }


ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
other_ = {'events': {'_av_trials': 'table'},'frontCam':{'camera':['times','ROIMotionEnergy']}}

session_types = ['multiSpaceWorld','postactive']
psths = []


type = 'aud'
for sess in session_types:
    session = { 
        'subject':mname,
        'expDate': expDate,
        'expDef': sess,
        'probe': probe
    }
    recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**session)
    if recordings.shape[0] == 1:            
        recordings =  recordings.iloc[0]
    else:
        print('recordings are ambiguously defined. Please recall.')


    events,spikes,clusters,_,cam = simplify_recdat(recordings,probe='probe')
    azimuths = [-60,60]
    for azi in azimuths:
        if 'aud' in type:
            if 'multiSpaceWorld' in recordings.expDef:
                to_keep_trials = (events.is_auditoryTrial & (events.stim_audAzimuth==azi) & (events.stim_audAmplitude==0.1) & events.is_validTrial & (events.first_move_time>raster_kwargs['post_time'])) 
            else:
                to_keep_trials = (events.is_auditoryTrial & (events.stim_audAzimuth==azi) & (events.stim_audAmplitude==0.1))
            t_on = events.timeline_audPeriodOn[to_keep_trials]

        if 'vis' in type:
            if 'multiSpaceWorld' in recordings.expDef:
                to_keep_trials = (events.is_coherentTrial & (events.stim_visAzimuth==azi) & (events.stim_audAmplitude==0.1) & (events.stim_visContrast==0.2) & events.is_validTrial) 
                n_ms = to_keep_trials.sum()
            else:
                to_keep_trials = (events.is_coherentTrial & (events.stim_visAzimuth==azi) & (events.stim_audAmplitude==0.1) & (events.stim_visContrast==0.2))           
            t_on = events.timeline_visPeriodOn[to_keep_trials]
            t_on = t_on[:n_ms]

        clus_ids = recordings.probe.clusters._av_IDs.astype('int') 

        r = get_binned_rasters(spikes.times,spikes.clusters,clus_ids,t_on,**raster_kwargs)
        # # so we need to break here so that we perorm getting binned rasters only once...
        #stim = r.rasters[:,:,r.tscale>=0]
        psth = r.rasters.mean(axis=0)
        print(sess)
        psths.append(psth[np.newaxis,:,:])

psths = np.concatenate(psths)



# %% 
# plot things just to check 
nID=208
nID = np.where(clus_ids==nID)[0][0]
color_list = ['blue','red','blue','red']
linestype_list = ['--','--','-','-']
for p in range(psths.shape[0]): 
    plt.plot(r.tscale,psths[p,nID,:],color=color_list[p],linestyle=linestype_list[p])
# %%
plt.plot(np.abs(psths[0,nID,:]-psths[1,nID,:]))
plt.plot(np.abs(psths[2,nID,:]-psths[3,nID,:]))

#%%
passive = np.max(np.abs(psths[0,:,:]-psths[1,:,:]),axis=1)

active = np.max(np.abs(psths[2,:,:]-psths[3,:,:]),axis=1)

# %%
is_good  = clusters.missed_spikes_est < 0.05
plt.plot(passive[is_good],active[is_good],'o')
plt.plot([0,60],[0,60],'k--')
plt.xlabel('passive max|audL-audR|')
plt.ylabel('active max|audL-audR|')


# %%
cID = clus_ids[is_good]
cID[np.argsort(passive[is_good]-active[is_good])]
# %%
