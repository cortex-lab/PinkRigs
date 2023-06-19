# this code is desinged to assess whethe the 'right' scale of power to use for each hemisphere
# %%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch

my_subject = 'AV041'
recordings = load_data(
    subject = my_subject,
    expDate = '2023-06-13',
    data_name_dict={'events':{'_av_trials':'all'}}
    )

# %% 
ev,_,_,_,_ = zip(*[simplify_recdat(rec) for _,rec in recordings.iterrows()])
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
            

# %% 
#ev,_,_,_ = simplify_recdat(recordings.iloc[0]) # write a merging procedure when the time comes
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
contrasts=contrasts[[1,3,5]]
# %%
choices = ev_.timeline_choiceMoveDir-1
power_options = np.array([-15,-10,-5,-2,0,2,5,10,15])
power_colors = plt.cm.coolwarm(np.linspace(0,1,len(power_options)))
_,keep_idx,_ = np.intersect1d(power_options,powers,return_indices=True)
power_colors = power_colors[keep_idx]

pR = [[np.nanmean(choices[(laser_power==p) & (stim_contrast==c)])for c in contrasts] for p in powers]
n_trials = [[(choices[(laser_power==p) & (stim_contrast==c)]).size for c in contrasts] for p in powers]
fig,ax = plt.subplots(1,1,figsize=(5,5))
fig.patch.set_facecolor('xkcd:white')
for p_idx in range(len(pR)):
    ax.plot(contrasts,pR[p_idx],color=power_colors[p_idx],label= '%.0d mW,n=%.0d' % (powers[p_idx],sum(n_trials[p_idx])))
fig.legend(loc=(0.5,0.005),bbox_to_anchor=(1.001, 1))
ax.set_xlabel('vis contrast')
ax.set_ylabel('p(Right)')
ax.set_ylim([0,1.05])
ax.axhline(0.5,xmin=0,xmax=1,color='k',linestyle='--',alpha=.2)
ax.axvline(0,ymin=0,ymax=1,color='k',linestyle='--',alpha=.2)
ax.set_title(my_subject)
# %%


