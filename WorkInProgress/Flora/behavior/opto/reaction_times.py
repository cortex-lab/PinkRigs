# %%

# %%
import sys
import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_topspines
from Analysis.pyutils.ev_dat import getTrialNames

my_subject = ['AV036']
recordings = load_data(
    subject = my_subject,
    expDate = '2022-05-02:2023-09-04',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )

# %% 
ev,_,_,_,_ = zip(*[simplify_recdat(rec,reverse_opto=True) for _,rec in recordings.iterrows()])

# %%
is_laser_session = [(np.sum(e.is_laserTrial)>0)  & (np.abs(e.stim_laserPosition)==1).any() for e in ev]
ev = list(compress(ev,is_laser_session))
# %%
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})
            
# %%
# calculate nogo per contrast and nogo per type 
azimuths = ev.stim_visAzimuth
azimuths[np.isnan(azimuths)]=0
ev.signed_contrast = ev.stim_visContrast * np.sign(azimuths)
contrasts = np.unique(ev.signed_contrast[~np.isnan(ev.signed_contrast)])

window_size = 1
n_trials = ev.timeline_choiceMoveDir.size
is_nogo = np.isnan(ev.response_direction==0)

indices = np.arange(0,n_trials-window_size)+window_size
is_in_nogo_block = [is_nogo[t-window_size:t].sum() for t in indices]
is_in_nogo_block = np.concatenate([np.zeros(window_size),np.array(is_in_nogo_block)])
is_in_nogo_block = (is_in_nogo_block>=window_size) & is_nogo

# %%
# calculate how many trials we have of each trialtype

nT = np.sum(ev.is_validTrial & ev.is_laserTrial & ((ev.stim_laserPosition)==0))
print(nT)

# %%
# plot performance 
aud_azimuths = np.unique(ev.stim_audAzimuth[~np.isnan(ev.stim_audAzimuth)])
azimuth_colors = plt.cm.coolwarm(np.linspace(0,1,aud_azimuths.size))


laser_keep_set = (ev.laser_power==17) & (np.abs(ev.stim_laserPosition)==1) & ev.is_validTrial & ev.is_laserTrial
fig,ax = plt.subplots(1,1,figsize=(5,5))
for i,a in enumerate(aud_azimuths):
    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ~ev.is_laserTrial 
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(r[~np.isnan(r)]) for r in rt_per_c])
    ax.plot(contrasts,np.log10(frac_no_go/(1-frac_no_go)),'o-',color=azimuth_colors[i])  

    to_keep_trials = laser_keep_set & (ev.stim_audAzimuth==a) 
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(r[~np.isnan(r)]) for r in rt_per_c])
    ax.plot(contrasts,np.log10(frac_no_go/(1-frac_no_go)),'*--',color=azimuth_colors[i])  

ax.set_ylabel('log[p(ipsi)/p(contra)]')
ax.set_xlabel('contrasts')
ax.set_title('%s, %.0d opto trials' % (my_subject,np.sum(laser_keep_set)))

# %% non log plot
fig,ax = plt.subplots(1,1,figsize=(5,5))

laser_keep_set = (ev.laser_power==17) & (((ev.stim_laserPosition))==-1) & ev.is_validTrial & ev.is_laserTrial 
for i,a in enumerate(aud_azimuths):
    to_keep_trials = laser_keep_set & (ev.stim_audAzimuth==a)
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(r[~np.isnan(r)]) for r in rt_per_c])
    ax.plot(contrasts,frac_no_go,'*--',color=azimuth_colors[i])  

    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ~ev.is_laserTrial 
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(r[~np.isnan(r)]) for r in rt_per_c])
    ax.plot(contrasts,frac_no_go,'o-',color=azimuth_colors[i])  

ax.set_xlabel('contrasts')
ax.set_title('%s, %.0d opto trials' % (my_subject,np.sum(laser_keep_set)))

# %% plot fraction of nogo 
fig,ax = plt.subplots(1,1,figsize=(5,5))
laser_keep_set = (ev.laser_power==10) & ((ev.stim_laserPosition)==1) & ev.is_validTrial & ev.is_laserTrial & ~is_in_nogo_block


d=pd.DataFrame()
for i,a in enumerate(aud_azimuths):
    to_keep_trials = laser_keep_set & (ev.stim_audAzimuth==a)
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(np.isnan(r)) for r in rt_per_c])
    frac_right= np.array([np.mean((r==1)) for r in rt_per_c])
    ax.plot(contrasts,frac_no_go,'*--',color=azimuth_colors[i])  

    d = d.append(pd.DataFrame({'nogo':frac_no_go,
                'ipsi':frac_right, 
                'contra':1-frac_right-frac_no_go,
                'contrasts':contrasts,
                'aud_azimuths':a,
                'opto':1}),ignore_index=True)

    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ~ev.is_laserTrial & ~is_in_nogo_block
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(np.isnan(r)) for r in rt_per_c])
    frac_right= np.array([np.mean((r==1)) for r in rt_per_c])

    d = d.append(pd.DataFrame({'nogo':frac_no_go,
                   'ipsi':frac_right, 
                   'contra':1-frac_right-frac_no_go,
                   'contrasts':contrasts,
                   'aud_azimuths':a,
                   'opto':0}),ignore_index=True)

    ax.plot(contrasts,frac_no_go,'o-',color=azimuth_colors[i])  

ax.set_ylabel('p(nogo)')
ax.set_xlabel('contrasts')
ax.set_title('%s, %.0d opto trials' % (my_subject,np.sum(laser_keep_set)))

# %%
# plot ternary plots
import plotly_express as px
import pandas as pd

d['contrasts_'] =np.abs(d.contrasts)
px.scatter_ternary(d,a='nogo',b='ipsi',c='contra',color='aud_azimuths',
                   symbol = 'opto',size='contrasts_',
                   color_continuous_scale=['blue','grey','red'],
                   symbol_sequence = ['x','circle']
                   )


# %%
# reaction times for left vs right choices on opto vs non-opto trials 
powers = np.unique(ev.laser_power)
fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

powers = np.array([0,5,10,17])
power_colors = plt.cm.inferno(np.linspace(.2,.8,powers.size))
for i,p in enumerate(powers): 
    to_keep_trials = ev.is_validTrial & (ev.laser_power==p) & (np.abs(ev.stim_laserPosition)!=0) #& ev.is_blankTrial
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    
    ax[0].hist(ev_.rt[ev_.timeline_choiceMoveDir==1],color=power_colors[i],bins=250,alpha=0.5,density=True,histtype='step',cumulative=True)#,cumulative=True,alpha=0.5,density=True)
    ax[1].hist(ev_.rt[ev_.timeline_choiceMoveDir==2],color=power_colors[i],bins=250,alpha=0.5,density=True,histtype='step',cumulative=True)# ,cumulative=True,alpha=0.5,density=True)

    ax[0].set_title('contra choices')
    ax[1].set_title('ipsi choices')
    off_topspines(ax[0])
    off_topspines(ax[1])
ax[1].legend(powers)


# %%


fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

powers = np.array([0,10])
power_colors = plt.cm.inferno(np.linspace(.2,.8,powers.size))
for i,p in enumerate(powers): 
    to_keep_trials = ev.is_validTrial & (ev.laser_power==p) & (np.abs(ev.stim_laserPosition)!=0) #& ev.is_blankTrial
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    
    #ax[0].hist(ev_.rt[ev_.timeline_choiceMoveDir==1],color=power_colors[i],bins=250,alpha=0.5,density=True,histtype='step',cumulative=True)#,cumulative=True,alpha=0.5,density=True)
    #ax[1].hist(ev_.rt[ev_.timeline_choiceMoveDir==2],color=power_colors[i],bins=250,alpha=0.5,density=True,histtype='step',cumulative=True)# ,cumulative=True,alpha=0.5,density=True)
    h,bins = np.histogram(ev_.rt[ev_.timeline_choiceMoveDir==1],bins=150)
    ax[0].plot(bins[1:],h/np.max(h),color=power_colors[i])
    h,bins = np.histogram(ev_.rt[ev_.timeline_choiceMoveDir==2],bins=150)
    ax[1].plot(bins[1:],h/np.max(h),color=power_colors[i])

    ax[0].set_title('contra choices')
    ax[1].set_title('ipsi choices')
    off_topspines(ax[0])
    off_topspines(ax[1])
ax[1].legend(powers)
plt.show()
# %%
# look at nogos per trial type

# separate reaction times by trial types 
import pandas as pd
import seaborn as sns

to_keep_trials = (ev.is_validTrial &
                  (((np.abs(ev.stim_laserPosition))==1)|(np.isnan(ev.stim_laserPosition))) & 
                  (ev.laser_power==34)|(ev.laser_power==0)|(ev.laser_power==17)|(ev.laser_power==10))

ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
df = pd.DataFrame(ev_)
df['trialNames'] = getTrialNames(ev_)

choice_signed = np.sign(ev_.timeline_choiceMoveDir-1.5)
df['powerXchoiceDir'] = (ev_.laser_power+1e-10) * choice_signed
# make a single array of trialtypes
df['rt_laser'] = ev_.block_firstMovePostLaserOn-ev_.timeline_audPeriodOn

fig,ax = plt.subplots(1,1,figsize=(7,13))
fig.patch.set_facecolor('xkcd:white')

sns.violinplot(data=df, 
              x="rt",
              y="trialNames", order = ['blank','auditory','visual','coherent','conflict'],
              hue="powerXchoiceDir", dodge=True, linewidth=.5,
              orient='h', trust_alpha=0.5, saturation=1,
              palette = "coolwarm",outlier_prop=0.00001,showfliers=False, width=0.7)

ax.set_ylabel('reaction time (s)')
#plt.legend([],[], frameon=False)
off_topspines(ax)
# mypath = r'C:\Users\Flora\Pictures\LakeConf'
# savename = mypath + '\\' + 'optoRTs.svg'

# fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%

import joypy
from matplotlib import cm
#
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Calibri Light'})
labels=[-17,-10,-0,0,10,17]
trialtypes = ['blank','auditory','visual','coherent','conflict']
fig,ax = plt.subplots(1,1,figsize=(12,8))
joypy.joyplot(data=df[df.rt<.8],column='rt',by='powerXchoiceDir',labels=labels,colormap=cm.coolwarm,ax=ax,kind='normalized_counts',bins=200,lw=3)
ax.set_xlabel('rt from stimulus (s)')
ax.set_xlim([0,1])
plt.show()
# %%
# summary plot of this matter would include (according to Pip)
# for 10mW & 17 mW
laser_keep_set = (ev.laser_power==10) & (np.abs(ev.stim_laserPosition)==1) & ev.is_validTrial & ev.is_laserTrial
 
for i,a in enumerate(aud_azimuths):
    to_keep_trials =  laser_keep_set & (ev.stim_audAzimuth==a)
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.rt[ev_.signed_contrast==c] for c in contrasts] 


print('sg')
# %%
