# %%
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_active_and_passive
from Analysis.pyutils.ev_dat import index_trialtype_perazimuth

session = { 
    'subject':'AV034',
    'expDate': '2022-12-07',
    'probe': 'probe0'
}
sessName = '%s_%s_%s' % tuple(session.values())
dat = load_active_and_passive(session)

# %%
from Analysis.pyutils.plotting import my_rasterPSTH,off_axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_raster_withmovementonset(spikes,leftchoicestims,rightchoicestims,choicemove_times,neuronID,ax,t_before=.2,t_after=1):
    
    plot_kwargs = {
        'pre_time':t_before,
        'post_time':t_after,
        'include_PSTH':False,
        'include_raster':True,
        'n_rasters':leftchoicestims.size+rightchoicestims.size,
        'ax':ax,
        'ax1':ax
    }  
    
    my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftchoicestims,
                    rightchoicestims],
                    neuronID,  # Identity of the cluster we plot 
                    event_colors=['blue','red'],rasterlw=1,**plot_kwargs)

    
    my_rasterPSTH(choicemove_times,  # Spike times first
                    np.ones(choicemove_times.size).astype('int'),  # Then cluster ids
                    [leftchoicestims,
                    rightchoicestims],
                    1,  # Identity of the cluster we plot,
                    event_colors=['black','black'],rasterlw=3,**plot_kwargs)
    
    off_axes(ax) 
    ax.set_ylim([-1000,10])

    return plot_kwargs



def plot_shadlen(spikes,leftstim,rightstim,leftmove,rightmove,nrnID,ax,ax1,plot_kwargs=None):
    #spec = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=mygs) #
    #ax=figID.add_subplot(spec[0])
    #ax1=figID.add_subplot(spec[1],sharey=ax) 

    #ax.set_ylim([0,40])
    #ax1.set_ylim([0,200])

    trial_cutoff=4

    if not plot_kwargs:
        plot_kwargs = {
            'pre_time':.2,
            'post_time':.3,
            'include_PSTH':True,
            'include_raster':False, 
            'n_rasters':leftstim.size+rightstim.size,
            'bin_size':.001,
            'smoothing':.025,
            'event_colors':['blue','red']
        }   

    if (leftstim.size>trial_cutoff) & (rightstim.size>trial_cutoff):
        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftstim,rightstim],
                    nrnID,  # Identity of the cluster we plot 
                    ax=ax,**plot_kwargs)

        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftmove,rightmove],
                    nrnID,  # Identity of the cluster we plot                 
                    ax=ax1,**plot_kwargs)

    elif (leftstim.size>trial_cutoff) & (rightstim.size<=trial_cutoff):
        plot_kwargs['event_colors']=['blue']

        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftstim],
                    nrnID,  # Identity of the cluster we plot 
                    ax=ax,**plot_kwargs)

        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [leftmove],
                    nrnID,  # Identity of the cluster we plot                 
                    ax=ax1,**plot_kwargs)

    elif (leftstim.size<=trial_cutoff) & (rightstim.size>trial_cutoff):
        plot_kwargs['event_colors']=['red']

        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [rightstim],
                    nrnID,  # Identity of the cluster we plot 
                    ax=ax,**plot_kwargs)

        my_rasterPSTH(spikes.times,  # Spike times first
                    spikes.clusters,  # Then cluster ids
                    [rightmove],
                    nrnID,  # Identity of the cluster we plot                 
                    ax=ax1,**plot_kwargs)

    elif (leftstim.size<=trial_cutoff) & (rightstim.size<=trial_cutoff):
        print('not enough trials for a psth')

    off_axes(ax) 
    off_axes(ax1) 

    return plot_kwargs

### some minor helper functions 
def visazimuth_check(ev,v,trialtype):
    if 'aud' in trialtype:
        visazimcheck =np.isnan(ev.stim_visAzimuth)
    elif 'blank' in trialtype:
        visazimcheck =np.isnan(ev.stim_visAzimuth)
    else:
        visazimcheck = (ev.stim_visAzimuth==v)
    
    return visazimcheck

# %%
nrnID = 118
spk = dat.multiSpaceWorld.spikes
ev = dat.multiSpaceWorld.events

movesort = True
plot_type = 'psth'
plot_passive = True


plotted_vis_azimuth = [-60,0,60]
plotted_aud_azimuth = [-60,0,60]
viscontrasts = (np.unique(ev.stim_visContrast)*100).astype('int')
viscontrasts = viscontrasts[viscontrasts>0]
viscontrasts = viscontrasts[-2]


viscontrasts = np.append(0,viscontrasts)
n_contrasts = (viscontrasts>0).sum()

plotted_vis_azimuth = np.append(np.append(np.ones(n_contrasts)*(-60),[0]),np.ones(n_contrasts)*(60)).astype('int')
plotted_vis_contrasts = np.abs(np.unique(np.array([viscontrasts,viscontrasts*-1])))
vazi,aazi=np.meshgrid(plotted_vis_azimuth,plotted_aud_azimuth)

if 'raster' in plot_type:
    fig,ax=plt.subplots(vazi.shape[0],vazi.shape[1],figsize=(15,15),sharex=True)
elif 'psth' in plot_type: 
    fig= plt.figure(figsize=(8, 8))
    #gs0 = gridspec.GridSpec(vazi.shape[0],vazi.shape[1]*2)
    fig,ax=plt.subplots(vazi.shape[0],vazi.shape[1]*2,figsize=(10,10),sharex=True,sharey=True)


fig.patch.set_facecolor('xkcd:white')
fig.tight_layout()
rt = ev.timeline_choiceMoveOn - np.nanmin(np.concatenate([ev.timeline_audPeriodOn[:,np.newaxis],ev.timeline_visPeriodOn[:,np.newaxis]],axis=1),axis=1)
for i,m in enumerate(vazi):
    for j,_ in enumerate(m):
        v = vazi[i,j]
        a = aazi[i,j]
        trialtype=index_trialtype_perazimuth(a,v)
        
        # check visual azimuths for 
        #visazimcheck = visazimuth_check(ev,v,trialtype)
        
        leftmove_idx = (ev.stim_audAzimuth==a) & visazimuth_check(ev,v,trialtype) & ((ev.stim_visContrast*100).astype('int')==plotted_vis_contrasts[j]) & (ev.timeline_choiceMoveDir==1) & (ev[trialtype]==1)
        rightmove_idx = (ev.stim_audAzimuth==a) & visazimuth_check(ev,v,trialtype)  & ((ev.stim_visContrast*100).astype('int')==plotted_vis_contrasts[j]) & (ev.timeline_choiceMoveDir==2) & (ev[trialtype]==1)

        passive_idx  = (
            (dat.postactive.events.stim_audAzimuth==a) & 
            visazimuth_check(dat.postactive.events,v,trialtype) & 
            (dat.postactive.events.stim_visContrast==plotted_vis_contrasts[j]/100)
        )

        leftstims = ev['timeline_audPeriodOn'][leftmove_idx]
        rightstims = ev['timeline_audPeriodOn'][rightmove_idx]
        passive_stims = dat.postactive.events.timeline_audPeriodOn[passive_idx]

        leftmove = ev.timeline_choiceMoveOn[leftmove_idx]
        rightmove = ev.timeline_choiceMoveOn[rightmove_idx]

        if movesort: 
            leftstims = leftstims[np.argsort(leftmove-leftstims)]
            rightstims = rightstims[np.argsort(rightmove-rightstims)]

        if 'raster' in plot_type:            

            raster_kwargs = plot_raster_withmovementonset(spk,
                                    leftstims,
                                    rightstims,ev.timeline_choiceMoveOn,
                                    [nrnID],ax[i,j])
            
            raster_kwargs['event_colors'] = ['grey']        
            my_rasterPSTH(dat.postactive.spikes.times,  # Spike times first
                    dat.postactive.spikes.clusters,  # Then cluster ids
                    [passive_stims],
                    nrnID,  # Identity of the cluster we plot 
                    ax=ax[i,j],**raster_kwargs)
            off_axes(ax[i,j]) 

        elif 'psth' in plot_type:
            psth_kwargs = plot_shadlen(spk,leftstims,rightstims,leftmove,rightmove,[nrnID],ax[i,j*2],ax[i,j*2+1])
            
            psth_kwargs['event_colors'] = ['grey']        
            my_rasterPSTH(dat.postactive.spikes.times,  # Spike times first
                    dat.postactive.spikes.clusters,  # Then cluster ids
                    [passive_stims],
                    nrnID,  # Identity of the cluster we plot 
                    ax=ax[i,j*2],**psth_kwargs)
            off_axes(ax[i,j*2])
            
name = sessName + '_neuron_%.0d' % nrnID         
fig.suptitle(name)

# %%
