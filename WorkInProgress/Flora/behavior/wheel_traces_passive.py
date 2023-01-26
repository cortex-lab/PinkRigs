# %%
import sys,glob
from pathlib import Path

pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\Audiovisual')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
import matplotlib.pyplot as plt
import numpy as np
from utils.io import add_github_paths
add_github_paths()

from Analysis.helpers.queryExp import load_data
subject = 'AV008'
data_dict = {
            'events':{'_av_trials':'all'}
                }
recordings = load_data(subject = subject, expDate = '2021-12-03',expDef = 'postactive',data_name_dict=data_dict)
# 
ev = recordings.iloc[0].events['_av_trials']

# 
def plot_mean_wheel(ev,selected_trials,t=None,ax=None,plot_mean = True,plot_all_trials = False, bl_subtract=True, **meankwargs):
    if not t: 
        t=np.arange(-0.1,1,0.001)
    # hacky way of subtracting the baseline
    bl_subtracted_wheelValue = np.array([i - i[0] for i in ev.timeline_wheelValue])
    wheelraster = np.interp(ev.timeline_audPeriodOn[selected_trials,np.newaxis]+t,
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
        ax.plot(t,meanwheel,**meankwargs,lw=2)
    if plot_all_trials:         
        [ax.plot(t,wheelraster[i,:]-(wheelraster[i,0:3]).mean(),**meankwargs,alpha=.2,lw=2) for i in range(wheelraster.shape[0])]

# %%
fig,ax = plt.subplots(1,1,figsize=(5,5))
selected = np.where(ev.stim_audAzimuth==-60)[0]

plot_mean_wheel(ev,selected,ax=ax,color='k',plot_mean = True,plot_all_trials=False)
ax.axvline(0,color='k')
# %%
# checking 