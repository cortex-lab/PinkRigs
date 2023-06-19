# %%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np

from Admin.csv_queryExp import load_data
subject = 'AV034'
data_dict = {
            'events':{'_av_trials':'all'}
                }
recordings = load_data(subject = subject, expDate = '2022-12-12',expDef = 'multiSpace',data_name_dict=data_dict)
# 
ev = recordings.iloc[0].events['_av_trials']
# %%
%matplotlib inline
is_go = ~np.isnan(ev.timeline_choiceMoveDir) 
reaction_times = ev.timeline_choiceMoveOn - ev.timeline_audPeriodOn

selected = np.where(is_go & (reaction_times>.05))[0]
# %%
# plot single trial 
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
i = selected[6]
ax.plot(ev.timeline_wheelTime[i],ev.timeline_wheelValue[i],color='k') 
ax.axvline(ev.timeline_audPeriodOn[i],color='r')
ax.axvline(ev.timeline_choiceMoveOn[i],color='g')
ax.axvline(ev.timeline_audPeriodOff[i],color='b')
fig.show()
# %%
# plot wheel trace starting at choice and baseline subtract
selected = np.where(is_go & (reaction_times>.05) & (ev.timeline_choiceMoveDir==2))[0]

fig,ax = plt.subplots(1,1,figsize=(5,5))

def plot_bl_subtracted_trace_singe(ev,i,ax,**kwargs):
    
    choice_move_idx = np.where(ev.timeline_wheelTime[i]==ev.timeline_choiceMoveOn[i])[0][0]
    ax.plot(ev.timeline_wheelTime[i][choice_move_idx:]-ev.timeline_audPeriodOn[i],ev.timeline_wheelValue[i][choice_move_idx:]-ev.timeline_wheelValue[i][choice_move_idx],**kwargs) 

[plot_bl_subtracted_trace_singe(ev,i,ax,color='k',alpha=.2) for i in selected]
# %%
# get the average wheel trace of selected_trials

def plot_mean_wheel(ev,selected_trials,t=None,ax=None,plot_mean = True,plot_all_trials = False, bl_subtract=True, **meankwargs):
    if not t: 
        t=np.arange(-0.1,.2,0.01)
    # hacky way of subtracting the baseline
    bl_subtracted_wheelValue = np.array([i - i[0] for i in ev.timeline_wheelValue])
    wheelraster = np.interp(ev.timeline_choiceMoveOn[selected_trials,np.newaxis]+t,
                            np.concatenate(ev.timeline_wheelTime[selected_trials]),
                            np.concatenate(bl_subtracted_wheelValue[selected_trials]))

    if bl_subtract: 
        zero_idx = np.argmin(np.abs(t))
        bl = np.nanmean(wheelraster[:,:zero_idx])
        bl = np.tile(bl,t.size)
        wheelraster = wheelraster - bl

    meanwheel = np.nanmean(wheelraster,axis=0)

    if not ax:
        _,ax = plt.subplots(1,1,figsize=(5,5))
    if plot_mean: 
        ax.plot(t,meanwheel,**meankwargs,lw=5)
    if plot_all_trials:         
        [ax.plot(t,wheelraster[i,:]-(wheelraster[i,0:3]).mean(),**meankwargs,alpha=.3,lw=2) for i in range(wheelraster.shape[0])]

# %% 
# plot left vs right 
fig,ax = plt.subplots(1,1,figsize=(5,5))

# fast trials 
selected_right = np.where(is_go & (reaction_times<.1) & (ev.timeline_choiceMoveDir==2))[0]
plot_mean_wheel(ev,selected_right,ax=ax,color='r',plot_all_trials=False)

selected_left = np.where(is_go & (reaction_times<.1) & (ev.timeline_choiceMoveDir==1))[0]
plot_mean_wheel(ev,selected_left,ax=ax,color='b',plot_all_trials=False)

# slower trials
selected_right = np.where(is_go & (reaction_times>.1) & (ev.timeline_choiceMoveDir==2))[0]
plot_mean_wheel(ev,selected_right,ax=ax,color='r',plot_all_trials=False,linestyle='dashed')

selected_left = np.where(is_go & (reaction_times>.1) & (ev.timeline_choiceMoveDir==1))[0]
plot_mean_wheel(ev,selected_left,ax=ax,color='b',plot_all_trials=False,linestyle='dashed')
ax.axvline(0,color='k')


# %%
# now I will do different speeds of turning towards the right
fig,ax = plt.subplots(1,1,figsize=(5,5))

vmins = np.array([.1,0.15,.2,.25]) 
vmaxs = vmins +.05

color_r = plt.cm.Reds(np.linspace(0.2,1,vmins.size))
color_l = plt.cm.Blues(np.linspace(0.2,1,vmins.size))

for idx,v in enumerate(zip(vmins,vmaxs,color_l,color_r)): 
    selected_r = np.where(is_go & (reaction_times>v[0])  & (reaction_times<v[1])& (ev.timeline_choiceMoveDir==2))[0]
    selected_l = np.where(is_go & (reaction_times>v[0])  & (reaction_times<v[1])& (ev.timeline_choiceMoveDir==1))[0]

    if selected_l.size>0: 
        plot_mean_wheel(ev,selected_l,ax=ax,color=v[2],plot_mean = True, plot_all_trials = True ,label='left,%.2f - %.2f s RT' %(v[0],v[1])) 
    if selected_r.size>0:
        plot_mean_wheel(ev,selected_r,ax=ax,color=v[3],plot_mean = True,plot_all_trials= True,label='right,%.2f - %.2f s RT' %(v[0],v[1]))

    
    ax.axvline(0,color='k')
    ax.axhline(0,color='k',linestyle='dashed',alpha=.5)

fig.legend(loc='center right',bbox_to_anchor=(1.4,.5))
fig.show()#
#ax.set_ylim([-60,60])
# %%
fig,ax = plt.subplots(1,1)
is_go = ~np.isnan(ev.timeline_choiceMoveDir)  & (ev.is_visualTrial) & (ev.is_validTrial)
selected_right = np.where(is_go & (ev.timeline_choiceMoveDir==2))[0]
ax.hist(reaction_times[selected_right],bins=30,color='red',alpha=.5)
selected_left = np.where(is_go & (ev.timeline_choiceMoveDir==1))[0]
ax.hist(reaction_times[selected_left],bins=30,color='b',alpha=.5)
ax.set_xlabel('reaction time (s)')

# %%
# chronometric curves 
contrasts = np.sign(ev.stim_visAzimuth) * ev.stim_visContrast
contrasts[np.isnan(contrasts)]=0
c_set = np.unique(contrasts)
rt_vis = [np.nanmedian(reaction_times[ev.is_validTrial & ev.is_visualTrial & (contrasts==c)])  for c in c_set]
rt_coh = [np.nanmedian(reaction_times[ev.is_validTrial & ev.is_coherentTrial & (contrasts==c)])  for c in c_set]
rt_conf = [np.nanmedian(reaction_times[ev.is_validTrial & ev.is_conflictTrial & (contrasts==c)])  for c in c_set]
rt_blank = [np.nanmedian(reaction_times[ev.is_validTrial & ev.is_blankTrial & (contrasts==c)])  for c in c_set]

rt_aud = [np.nanmedian(reaction_times[ev.is_validTrial & ev.is_auditoryTrial & (np.sign(ev.stim_audAzimuth)==c)]) for c in np.sign(c_set)]

plt.plot(c_set,rt_vis,'b')
plt.plot(c_set,rt_coh,'r')
plt.plot(c_set,rt_conf,'g')
plt.plot(c_set,rt_blank,'ko')
plt.plot(c_set,rt_aud,'m--')
# %%
