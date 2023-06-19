# %%
# general loading functions
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

# built-in modules
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# Figure 1B - example visual neuron
from Admin.csv_queryExp import load_data,simplify_recdat,load_ephys_independent_probes
from Analysis.pyutils.ev_dat import postactive
from Analysis.pyutils.plotting import my_rasterPSTH,off_axes

# load single dataset 


column_names = ['subject','expDate','expNum','probe']
rec = [('AV025','2022-11-07',4,'probe0')]
cID = 34
rec = pd.DataFrame(rec,columns= column_names)
 #%%
 ############ example rasters ################

probe =  rec.iloc[0].probe   
recordings = load_data(
    data_name_dict={
        'events':{'_av_trials':'table'},
        probe:{'spikes':['times','clusters']}
    },
    **rec.iloc[0,:3]
    )

ev,spikes,_,_,_ = simplify_recdat(recordings.iloc[0],probe=probe)
b,v,a,_ = postactive(ev)
# %%
azimuths =np.array([-90,-60,-30,0,30,60,90])  # which azimuths to plot 
sel_contrast = v.contrast.max().values
sel_spl = a.SPL.max().values

# parameters of plotting 
bin_kwargs={'tscale':[None],
            'pre_time':.3,'post_time': .5, 
            'bin_size':0.005, 'smoothing':0.02,
            'return_fr':True,'baseline_subtract':True
            }

event_kwargs = {
        'event_colors':['blue','magenta']
}

plot_kwargs = {
        'pethlw':2, 'rasterlw':2, 
        'erralpha':.4, 
        'n_rasters':30,
        'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'grey',

}
# 
fig,ax=plt.subplots(1,azimuths.size,figsize=(14,2),sharey=True)
for idx,azi in enumerate(azimuths):
    VisOnsets = v.sel(azimuths=azi,contrast=sel_contrast).values.flatten()
    AudOnsets = a.sel(azimuths=azi,SPL=sel_spl).values.flatten()

    my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets, AudOnsets],
                    [cID],ax=ax[idx],ax1=ax[idx],include_PSTH=False,include_raster=True,
                    **bin_kwargs,**plot_kwargs,**event_kwargs)

    ax[idx].set_xlabel('%.0f deg' % azi)
    off_axes(ax[idx])
    if idx==azimuths.size-1:
        plt.hlines(-600,0.2,0.4,'k',lw=3)

plt.suptitle(cID)
plt.show()
sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
nrn_name = '_av_IDs_%.0f' % cID
mypath = r'C:\Users\Flora\Pictures\LakeConf'
savename = mypath + '\\' + sessName + nrn_name + '.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
#################### EXAMPLE TUNING CURVES  ######################
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning

azi = azimuthal_tuning(rec.to_dict('r')[0])
#cID = 325

tuning_type = 'vis'
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': 0.02, # means I select the max
    'which': tuning_type,
    'subselect_neurons':None,
    'trim_type': None, 
    'trim_fraction':None
}

azi.get_rasters_perAzi(**tuning_curve_params)
tuning_curves,is_selective = azi.get_significant_fits(curve_type= 'gaussian',metric='svd')

azi.plot_response_per_azimuth(neuronID=cID,which='p')
fig = azi.plot_tuning_curves(tuning_curves=tuning_curves,neuronID=cID,metric='svd',plot_trials=False)

sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
nrn_name = '_av_IDs_%.0f' % cID
mypath = r'C:\Users\Flora\Pictures\LakeConf'
savename = mypath + '\\' + sessName + nrn_name + 'tuning_curve.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
####################### EXAMPLE MOVEMENT CORRELATIONG NEURONS  ###############
ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'_av_IDs'}
other_ = {'events': {'_av_trials': 'table'},'frontCam':{'camera':['times','ROIMotionEnergy']}}
import scipy
recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**rec.iloc[0])
recordings = recordings.iloc[0]
from Analysis.neural.utils.spike_dat import bincount2D
_,spikes,_,_,cam = simplify_recdat(recordings,probe='probe')
clus_ids = recordings.probe.clusters._av_IDs.astype('int') 
r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin=0.1)
x,y = cam.times,cam.ROIMotionEnergy
interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
camtrace = interp_func(t_bins[t_bins<x[-1]])
cID=118
fig,ax = plt.subplots(2,1,sharex=True)
pre,post = 300,500
ax[0].plot(t_bins[pre:post],camtrace[pre:post],'k')
spiketrace = r[np.where(clus==cID)[0][0],:]
ax[-1].hlines(-.01,t_bins[post]-1,t_bins[post],'k',lw=5)
ax[1].plot(t_bins[pre:post],spiketrace[pre:post],'grey')
off_axes(ax[0])
off_axes(ax[1])

sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
nrn_name = '_av_IDs_%.0f' % cID
mypath = r'C:\Users\Flora\Pictures\LakeConf'
savename = mypath + '\\' + sessName + nrn_name + 'movement_corr.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
#################### example mulitsensory interaction ########################

probe =  rec.iloc[0].probe   
recordings = load_data(
    data_name_dict={
        'events':{'_av_trials':'table'},
        probe:{'spikes':['times','clusters']}
    },
    **rec.iloc[0,:3]
    )

ev,spikes,_,_,_ = simplify_recdat(recordings.iloc[0],probe=probe)
b,v,a,ms = postactive(ev)


bin_kwargs={'tscale':[None],
            'pre_time':0.01,'post_time': .3, 
            'bin_size':0.005, 'smoothing':0.02,
            'return_fr':True,'baseline_subtract':True
            }

plot_kwargs = {
        'pethlw':2, 'rasterlw':2, 
        'erralpha':.4, 
        'n_rasters':30,
        'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'grey',

}
# 
fig,ax = plt.subplots(3,3,sharex=True,sharey=True)

vpref = 90
apref = 90 
sel_contrast = v.contrast.max().values
sel_spl = a.SPL.max().values

VisOnsets = v.sel(azimuths=-vpref,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets],
                [cID],ax=ax[2,1],ax1=ax[2,1],include_PSTH=True,include_raster=False,event_colors=['grey'],
                **bin_kwargs,**plot_kwargs)


VisOnsets = v.sel(azimuths=vpref,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets],
                [cID],ax=ax[2,2],ax1=ax[2,2],include_PSTH=True,include_raster=False,event_colors=['grey'],
                **bin_kwargs,**plot_kwargs)

Onsets = a.sel(azimuths=-apref,SPL=sel_spl).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[1,0],ax1=ax[1,0],include_PSTH=True,include_raster=False,event_colors=['grey'],
                **bin_kwargs,**plot_kwargs)

Onsets = a.sel(azimuths=apref,SPL=sel_spl).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[0,0],ax1=ax[0,0],include_PSTH=True,include_raster=False,event_colors=['grey'],
                **bin_kwargs,**plot_kwargs)


Onsets = ms.sel(visazimuths=vpref,audazimuths=apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[0,2],ax1=ax[0,2],include_PSTH=True,include_raster=False,event_colors=['green'],
                **bin_kwargs,**plot_kwargs)


Onsets = ms.sel(visazimuths=vpref,audazimuths=-apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[1,2],ax1=ax[1,2],include_PSTH=True,include_raster=False,event_colors=['thistle'],
                **bin_kwargs,**plot_kwargs)


Onsets = ms.sel(visazimuths=-vpref,audazimuths=apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[0,1],ax1=ax[0,1],include_PSTH=True,include_raster=False,event_colors=['paleturquoise'],
                **bin_kwargs,**plot_kwargs)

Onsets = ms.sel(visazimuths=-vpref,audazimuths=-apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
                [cID],ax=ax[1,1],ax1=ax[1,1],include_PSTH=True,include_raster=False,event_colors=['grey'],
                **bin_kwargs,**plot_kwargs)

for i in range(3):
    for j in range(3):
        off_axes(ax[i,j])


sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
nrn_name = '_av_IDs_%.0f' % cID
mypath = r'C:\Users\Flora\Pictures\LakeConf'
savename = mypath + '\\' + sessName + nrn_name + 'multisensory.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
