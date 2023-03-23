# this code is desinged to assess whethe the 'right' scale of power to use for each hemisphere
# %%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch

recordings = load_data(
    subject = 'AV038',
    expDate = '2023-03-15',
    data_name_dict={'events':{'_av_trials':'all'}}
    )

ev,_,_,_ = simplify_recdat(recordings.iloc[0]) # write a merging procedure when the time comes
# for unilateral inactivations laser power can be summed 
# do a sanity check that laser trials are only on visual trials

no_laserTrialtypes = (
    np.sum(ev.is_visualTrial[ev.is_laserTrial])+
    np.sum(ev.is_blankTrial[ev.is_laserTrial]) + 
    np.sum(ev.is_noStimTrial[ev.is_laserTrial])
)

is_only_vis = no_laserTrialtypes == sum(ev.is_laserTrial)
print('keeping vis trials only is:',is_only_vis)
# keep only blank and visual trials for ev 
to_keep_trials =(ev.is_visualTrial | ev.is_blankTrial) & ev.is_validTrial
ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
##
laser_power = (ev_.stim_laser1_power+ev_.stim_laser2_power)*ev_.stim_laserPosition
laser_power = laser_power.astype('int')
powers = np.unique(laser_power)

azimuths = ev_.stim_visAzimuth
azimuths[np.isnan(azimuths)]=0
stim_contrast = ev_.stim_visContrast * np.sign(azimuths)
contrasts = np.unique(stim_contrast)
# %%
choices = ev_.timeline_choiceMoveDir-1
power_colors = plt.cm.coolwarm(np.linspace(0,1,powers.size))

pR = [[np.nanmean(choices[(laser_power==p) & (stim_contrast==c)])for c in contrasts] for p in powers]
n_trials = [[(choices[(laser_power==p) & (stim_contrast==c)]).size for c in contrasts] for p in powers]
fig,ax = plt.subplots(1,1,figsize=(5,5))

for p_idx in range(len(pR)):
    ax.plot(contrasts,pR[p_idx],color=power_colors[p_idx],label= '%.0d mW,n=%.0d' % (powers[p_idx],sum(n_trials[p_idx])))
fig.legend()
ax.set_xlabel('vis contrast')
ax.set_ylabel('p(Right)')
# %%


