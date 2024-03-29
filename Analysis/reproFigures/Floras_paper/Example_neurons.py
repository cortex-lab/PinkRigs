# %%
# general loading functions
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

# built-in modules
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pathlib import Path

# Figure 1B - example visual neuron
from Admin.csv_queryExp import load_data,simplify_recdat,load_ephys_independent_probes
from Analysis.pyutils.ev_dat import postactive
from Analysis.pyutils.plotting import my_rasterPSTH,off_axes
from Analysis.pyutils.video_dat import get_move_raster

# load single dataset 


column_names = ['rd','subject','expDate','expNum','probeID']

#rec = pd.read_csv(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\postactiveSC.csv')
rec = [('rs','FT009','2021-01-20',8,'probe0')]
rec = pd.DataFrame(rec,columns= column_names)
 #%%
 ############ prepare the data ################
for rec_idx in range(len(rec)):
    try:
        probe =  rec.iloc[rec_idx].probeID   
        recordings = load_data(
            data_name_dict={
                'events':{'_av_trials':'table'},
                probe:{'spikes':['times','clusters'],'clusters':'_av_IDs'},
                # 'frontCam':{'camera':'all'},
                # 'sideCam':{'camera':'all'}

            },
            **rec.iloc[rec_idx,1:4]
            )

        ev,spikes,clusters,_,cam = simplify_recdat(recordings.iloc[0],probe=probe,cam_hierarchy=['sideCam','frontCam','eyeCam'])
        b,v,a,ms = postactive(ev)


        mypath = Path(r'D:\TempIms')

        save_format = 'svg'
#,,346
        for cID in [29,92,105,113]: #clusters._av_IDs:

            ############ rasters aligned to stimulus ############
            

            sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[rec_idx,1:])
            nrn_name = '_av_IDs_%.0f' % cID
            im_name = sessName + nrn_name + '.%s' % save_format

            azimuths =np.array([-90,-60,-30,0,30,60,90])  # which azimuths to plot 

            #azimuths = np.array([-60,0,60])
            sel_contrast = v.contrast.max().values
            sel_spl = a.SPL.max().values

            # parameters of plotting 
            bin_kwargs={'tscale':[None],
                        'pre_time':.1,'post_time': .2, 
                        'bin_size':0.005, 'smoothing':0.02,
                        'return_fr':True,'baseline_subtract':True
                        }

            event_kwargs = {
                    'event_colors':['blue','magenta']
            }

            plot_kwargs = {
                    'pethlw':2, 'rasterlw':1, 
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
                    plt.hlines(-600,0.1,0.2,'k',lw=3)

            plt.suptitle(cID)

            cpath = mypath / 'stimrasters'
            if not cpath.is_dir():
                cpath.mkdir(parents=True,exist_ok=True)

            savename = cpath / im_name
            fig.savefig(savename,transparent=False,bbox_inches = "tight",format=save_format,dpi=300)


            if cam is not None:
                ###################### SORTED MOVEMENT vs RASTER #################

                bin_kwargs  = {
                    'pre_time':.1,
                    'post_time':.25, 
                    'bin_size': 0.005
                }

                fig,ax = plt.subplots(1,2,figsize=(5,2.5),sharey=True)
                fig.patch.set_facecolor('xkcd:white')

                cam_values = (cam.ROIMotionEnergy)
                #cam_values = (cam_values-np.median(cam_values))/median_abs_deviation(cam_values)
                onset_times = ev.timeline_audPeriodOn[~np.isnan(ev.timeline_audPeriodOn)]#& (ev.stim_audAzimuth==60)]
                move_raster,_,sort_idx  = get_move_raster(onset_times,cam.times,cam_values,
                                                sortAmp=True,to_plot=True,**bin_kwargs,baseline_subtract=False,ax=ax[0])

                plot_kwargs = {
                        'pethlw':2, 'rasterlw':1.5, 
                        'erralpha':.4, 
                        'n_rasters':sort_idx.size,
                        'event_colors':['k'],
                        'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'red',

                }

                my_rasterPSTH(spikes.times,spikes.clusters,
                            [onset_times[sort_idx]],[cID], include_PSTH=False,reverse_raster=True,
                            **bin_kwargs,**plot_kwargs, ax = ax[1], ax1=ax[1])
                

                off_axes(ax[1])
                ax[1].hlines(onset_times.size*10+100,0.15,0.2,color='k')
                ax[1].vlines(-.1,onset_times.size*8-1000,onset_times.size*8,color='k')

                ax[0].hlines(onset_times.size*10+100,50,60,color='k')
                ax[0].vlines(-.1,onset_times.size*8-1000,onset_times.size*8,color='k')

                cpath = mypath / 'moverasters'
                if not cpath.is_dir():
                    cpath.mkdir(parents=True,exist_ok=True)
                    
                savename = cpath / im_name
                fig.savefig(savename,transparent=False,bbox_inches = "tight",format=save_format,dpi=100)

    except:
        print('did not work...')


# %% #################### EXAMPLE TUNING CURVES  ######################
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
column_names = ['subject','expDate','expNum','probe']

#rec = pd.read_csv(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\postactiveSC.csv')
rec = [('FT008','2021-01-15',5,'probe0')]
rec = pd.DataFrame(rec,columns= column_names)

azi = azimuthal_tuning(rec.to_dict('r')[0])
#cID = 325

tuning_type = 'aud'
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': None, # means I select the max
    'which': tuning_type,
    'subselect_neurons':None,
    'trim_type': None, 
    'trim_fraction':None
}

azi.get_rasters_perAzi(**tuning_curve_params)
tuning_curves,is_selective = azi.get_significant_fits(curve_type= 'gaussian',metric='svd')
#%%
cID=222
azi.plot_response_per_azimuth(neuronID=cID,which='p')
fig = azi.plot_tuning_curves(tuning_curves=tuning_curves,neuronID=cID,metric='svd',plot_trials=False)
fig.tight_layout()
sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
nrn_name = '_av_IDs_%.0f' % cID
mypath = r'C:\Users\Flora\Pictures\LakeConf'
savename = mypath + '\\' + sessName + nrn_name + 'tuning_curve.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%        ####################### EXAMPLE MOVEMENT CORRELATIONG NEURONS  ###############
# after we identified the neurons in Fig 3 I will come back to this 
# import scipy
# cID = 33
# #[76,15,89,193,49,4,35,62,99,167,357,281]:
# from Analysis.neural.utils.spike_dat import bincount2D
# clus_ids = clusters._av_IDs.astype('int') 
# r,t_bins,clus = bincount2D(spikes.times,spikes.clusters,xbin=0.1)
# x,y = cam.times,cam.ROIMotionEnergy
# interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
# camtrace = interp_func(t_bins[t_bins<x[-1]])
# fig,ax = plt.subplots(2,1,sharex=True)
# pre,post = 200,500
# ax[0].plot(t_bins[pre:post],camtrace[pre:post],'k')
# spiketrace = r[np.where(clus==cID)[0][0],:]
# ax[-1].hlines(-.01,t_bins[post]-1,t_bins[post],'k',lw=5)
# ax[1].plot(t_bins[pre:post],spiketrace[pre:post],'grey')
# off_axes(ax[0])
# off_axes(ax[1])

# sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[rec_idx,1:])
# nrn_name = '_av_IDs_%.0f' % cID
# mypath = r'C:\Users\Flora\Pictures\LakeConf'
# savename = mypath + '\\' + sessName + nrn_name + 'movement_corr.svg'
# fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# # # %%
# %% #################### example mulitsensory interaction ########################

# bin_kwargs={'tscale':[None],
#             'pre_time':0.01,'post_time': .3, 
#             'bin_size':0.005, 'smoothing':0.02,
#             'return_fr':True,'baseline_subtract':True
#             }

# plot_kwargs = {
#         'pethlw':2, 'rasterlw':2, 
#         'erralpha':.4, 
#         'n_rasters':30,
#         'onset_marker': 'tick','onset_marker_size':10,'onset_marker_color':'grey',

# }
# # 
# fig,ax = plt.subplots(3,3,sharex=True,sharey=True)

# vpref = 60
# apref = 60 
# sel_contrast = v.contrast.max().values
# sel_spl = a.SPL.max().values

# VisOnsets = v.sel(azimuths=-vpref,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets],
#                 [cID],ax=ax[2,1],ax1=ax[2,1],include_PSTH=True,include_raster=False,event_colors=['grey'],
#                 **bin_kwargs,**plot_kwargs)


# VisOnsets = v.sel(azimuths=vpref,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[VisOnsets],
#                 [cID],ax=ax[2,2],ax1=ax[2,2],include_PSTH=True,include_raster=False,event_colors=['grey'],
#                 **bin_kwargs,**plot_kwargs)

# Onsets = a.sel(azimuths=-apref,SPL=sel_spl).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[1,0],ax1=ax[1,0],include_PSTH=True,include_raster=False,event_colors=['grey'],
#                 **bin_kwargs,**plot_kwargs)

# Onsets = a.sel(azimuths=apref,SPL=sel_spl).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[0,0],ax1=ax[0,0],include_PSTH=True,include_raster=False,event_colors=['grey'],
#                 **bin_kwargs,**plot_kwargs)


# Onsets = ms.sel(visazimuths=vpref,audazimuths=apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[0,2],ax1=ax[0,2],include_PSTH=True,include_raster=False,event_colors=['green'],
#                 **bin_kwargs,**plot_kwargs)


# Onsets = ms.sel(visazimuths=vpref,audazimuths=-apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[1,2],ax1=ax[1,2],include_PSTH=True,include_raster=False,event_colors=['thistle'],
#                 **bin_kwargs,**plot_kwargs)


# Onsets = ms.sel(visazimuths=-vpref,audazimuths=apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[0,1],ax1=ax[0,1],include_PSTH=True,include_raster=False,event_colors=['paleturquoise'],
#                 **bin_kwargs,**plot_kwargs)

# Onsets = ms.sel(visazimuths=-vpref,audazimuths=-apref,SPL=sel_spl,contrast=sel_contrast).values.flatten()

# my_rasterPSTH(spikes.times,spikes.clusters,[Onsets],
#                 [cID],ax=ax[1,1],ax1=ax[1,1],include_PSTH=True,include_raster=False,event_colors=['grey'],
#                 **bin_kwargs,**plot_kwargs)

# for i in range(3):
#     for j in range(3):
#         off_axes(ax[i,j])


# sessName = '%s_%s_%.0f_%s' % tuple(rec.iloc[0])
# nrn_name = '_av_IDs_%.0f' % cID
# mypath = r'C:\Users\Flora\Pictures\LakeConf'
# savename = mypath + '\\' + sessName + nrn_name + 'multisensory.svg'
# fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
