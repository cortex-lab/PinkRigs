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

my_subject = ['AV046']
recordings = load_data(
    subject = my_subject,
    expDate = '2022-05-02:2023-08-31',
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
is_nogo = np.isnan(ev.timeline_choiceMoveDir)

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

plt.show()
# %% non log plot
fig,ax = plt.subplots(1,1,figsize=(5,5))

laser_keep_set = (ev.laser_power==17) & (np.abs((ev.stim_laserPosition))==1) & ev.is_validTrial & ev.is_laserTrial 
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
laser_keep_set = (ev.laser_power==17) & (np.abs(ev.stim_laserPosition)==1) & ev.is_validTrial & ev.is_laserTrial & ~is_in_nogo_block


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
laser_keep_set = (ev.laser_power==17) & (np.abs(ev.stim_laserPosition)==1) & ev.is_validTrial & ev.is_laserTrial & ~is_in_nogo_block

import ternary

scale = 1.0
figure, tax = ternary.figure(scale=scale)
# figure.set_size_inches(10, 10)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.gridlines(color="blue", multiple=.1)

# Set Axis labels and Title
fontsize = 20
offset = 0.2
tax.right_corner_label("pIpsi", fontsize=fontsize)
tax.top_corner_label("pNoGo", fontsize=fontsize)
tax.left_corner_label("pContra", fontsize=fontsize)


for i,a in enumerate(aud_azimuths):
    to_keep_trials = laser_keep_set & (ev.stim_audAzimuth==a)
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(np.isnan(r)) for r in rt_per_c])
    frac_right= np.array([np.mean((r==1)) for r in rt_per_c])

    d_opto = np.concatenate((frac_right[:,np.newaxis],frac_no_go[:,np.newaxis],(1-frac_right-frac_no_go)[:,np.newaxis]),axis=1)

    to_keep_trials = ev.is_validTrial & (ev.stim_audAzimuth==a) & ~ev.is_laserTrial & ~is_in_nogo_block
    ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})
    rt_per_c = [ev_.timeline_choiceMoveDir[ev_.signed_contrast==c]-1 for c in contrasts] 
    frac_no_go = np.array([np.mean(np.isnan(r)) for r in rt_per_c])
    frac_right= np.array([np.mean((r==1)) for r in rt_per_c])

    d_ctrl = np.concatenate((frac_right[:,np.newaxis],frac_no_go[:,np.newaxis],(1-frac_right-frac_no_go)[:,np.newaxis]),axis=1)


    for p1,p2 in zip(d_ctrl,d_opto):
    
        tax.line(p1, p2, linewidth=1., marker=None, color=azimuth_colors[i], linestyle="--")
        tax.scatter([p1], marker='.', color=azimuth_colors[i])

        tax.scatter([p2], marker='o', color=azimuth_colors[i])

tax.ticks(axis='lbr', multiple=1, linewidth=1, offset=0.025)
tax.get_axes().axis('off')

# %%
