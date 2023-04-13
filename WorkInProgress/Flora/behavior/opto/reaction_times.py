# %%

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_topspines
my_subject = 'AV036'
recordings = load_data(
    subject = my_subject,
    expDate = '2023-03-30:2023-04-12',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'all'}}
    )

# %% 
ev,_,_,_ = zip(*[simplify_recdat(rec) for _,rec in recordings.iterrows()])
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
            
# %%
# calculate nogo per contrast and nogo per type 
azimuths = ev.stim_visAzimuth
azimuths[np.isnan(azimuths)]=0
ev.signed_contrast = ev.stim_visContrast * np.sign(azimuths)
contrasts = np.unique(ev.signed_contrast)

# calculate reaction times 
ev.rt = ev.timeline_choiceMoveOn - np.nanmin(np.concatenate([ev.timeline_audPeriodOn[:,np.newaxis],ev.timeline_visPeriodOn[:,np.newaxis]],axis=1),axis=1)

# powers 
laser_power = (ev.stim_laser1_power+ev.stim_laser2_power)*ev.stim_laserPosition
ev.laser_power = laser_power.astype('int')
# %%
aud_azimuths = np.unique(ev.stim_audAzimuth[~np.isnan(ev.stim_audAzimuth)])
azimuth_colors = plt.cm.coolwarm(np.linspace(0,1,aud_azimuths.size))

fig,ax = plt.subplots(1,1,figsize=(5,5))
for i,a in enumerate(aud_azimuths):
    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ev.is_laserTrial
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.rt[ev_.signed_contrast==c] for c in contrasts] 
    frac_no_go = [np.mean(np.isnan(r)) for r in rt_per_c]
    ax.plot(contrasts,frac_no_go,'o-',color=azimuth_colors[i])

    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ~ev.is_laserTrial
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.rt[ev_.signed_contrast==c] for c in contrasts] 
    frac_no_go = [np.mean(np.isnan(r)) for r in rt_per_c]
    ax.plot(contrasts,frac_no_go,'--',color=azimuth_colors[i])  
ax.set_ylabel('p(no-go)')
ax.set_xlabel('contrasts')
# %%
# reaction times for left vs right choices on opto vs non-opto trials 
powers = np.unique(ev.laser_power)
power_colors = plt.cm.coolwarm(np.linspace(0,1,powers.size))
fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

for i,p in enumerate(powers): 
    to_keep_trials = ev.is_validTrial & (ev.laser_power==p)
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})


    ax[0].hist(ev_.rt[ev_.timeline_choiceMoveDir==1],color=power_colors[i],bins=100,alpha=0.5)#,cumulative=True,alpha=0.5,density=True)
    ax[1].hist(ev_.rt[ev_.timeline_choiceMoveDir==2],color=power_colors[i],bins=100,alpha=0.5)# ,cumulative=True,alpha=0.5,density=True)

    ax[0].set_title('left choices')
    ax[1].set_title('right choices')
    off_topspines(ax[0])
    off_topspines(ax[1])
ax[1].legend(powers)
# %%
# to unify all inhibitons, we reverse the conditions for left inhibition
# i.e. stimAzimuth and choices are reversed on those trials and then we don't 'sign' power (as it will always appear that inhibition was on the right)
# but to be fair we should do this session-by-session and not trial by trial to be fair....
ev_IC = ev
