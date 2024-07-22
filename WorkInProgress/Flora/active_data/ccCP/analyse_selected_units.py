# %%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.cccp import cccp,get_default_set
from Admin.csv_queryExp import load_data,format_cluster_data,bombcell_sort_units,get_subregions

pars = get_default_set(which='single_bin',t_length=0.2,t_bin=0.005)

# loading
mname = 'AV034'
expDate = '2022-12-10'
probe = 'probe0'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
}


ephys_dict = {'spikes':'all','clusters':'all'}
current_params = {
'data_name_dict':{'probe0':ephys_dict,'probe1':ephys_dict,
                    'events': {'_av_trials': 'table'}},
                    'checkSpikes':'1',
                    'unwrap_probes':True,
                    'filter_unique_shank_positions':True,
                    'merge_probes':False,                              
                    'cam_hierarchy': None

}  
recordings = load_data(**session,**current_params)
#%%
rec = recordings.iloc[0]

c = cccp()
c.load_and_format_data(rec=rec)

c.aud_azimuths=[0]
u,p,_,t = zip(*[c.get_U(which_dat='neural',**cp) for _,cp in pars.iterrows()])
pA,pC,pV,pB = p[0],p[1],p[2],p[3]
clusInfo = format_cluster_data(rec.probe.clusters)

thr = .05#/4
clusInfo['is_aud'] = np.concatenate(pA)<thr
clusInfo['is_choice'] = np.concatenate(pC)<thr
clusInfo['is_vis'] = np.concatenate(pV)<thr
clusInfo['is_choice_prestim'] = np.concatenate(pB)<thr

bc_class = bombcell_sort_units(clusInfo)#%%
clusInfo['is_good'] = (bc_class!='noise') #or (bc_class=='mua') 
clusInfo['Beryl'] = get_subregions(clusInfo.brainLocationAcronyms_ccf_2017.values,mode='Beryl')




#%%
# reID the neurons such that  
# all good,SC,
region = 'MRN'
type = 'choice'
goodIDs = clusInfo[
   clusInfo['is_%s' % type]
 & clusInfo.is_good 
   # & (clusInfo.Beryl==region)
    ]

# goodIDs = clusInfo[
#     clusInfo.is_good
# ]

goodIDs
# goodIDs.Beryl.hist()
# %%

# get the raster for each cell aligned to stimulus and choice (plot each cell)
from Analysis.pyutils.plotting import my_rasterPSTH,off_axes,off_excepty
# plot all cells in a matrix
import matplotlib.pyplot as pl


contrast = c.trial_types.contrast
visDiff = contrast * np.sign(c.trial_types.vis_azimuths)
choices = np.sign(c.trial_types.choice_type-1.5)
audPos = np.sign(c.trial_types.aud_azimuths)

# create the array of possibilities we want to plot
my_contrasts = np.sort(contrast.unique())+1e-3
contrast_options = np.ravel(np.sort(np.array([-my_contrasts,my_contrasts])))
my_choices = np.sign(contrast_options)
colors = pl.cm.coolwarm((contrast_options/max(contrast)+1)/2)




contrast_options = np.round(contrast_options,decimals=2)
pre_times = [0.1,0.1]
post_times =[0.4,0.1]

fig,ax = plt.subplots(2,7,figsize=(20,6),
                      gridspec_kw={
                          'width_ratios':[post_times[0],pre_times[1],post_times[0],pre_times[1],post_times[0],pre_times[1],0.02],
                          'height_ratios':[1,2], 
                          })

align_time_types = ['timeline_audPeriodOn',
              'timeline_choiceMoveOn']



ax[0,0].get_shared_y_axes().join(ax[0, 0], ax[0, 1],ax[0,2],ax[0,3],ax[0,4],ax[0,5])

all_aud_pos = [-1,0,1]


sel_spike_clusIDs = np.isin(c.spikes.clusters,goodIDs._av_IDs.values).astype('int')
for ai,a_ind in enumerate(all_aud_pos):
    req_a_pos = np.sign(contrast_options)*a_ind

    for i,align_time in enumerate(align_time_types):
        ev_list,color_list = [],[]

        for v,a,choice,my_color in zip(contrast_options,req_a_pos,my_choices,colors):
            trialID = np.where((visDiff==v) & (audPos==a) & (choices==choice))
            is_current_trial = c.ev.trial_type_IDs==trialID[0]

            if np.sum(is_current_trial)>0:
                rts = c.ev.rt[is_current_trial]
                ev_list.append(
                    c.ev[align_time][is_current_trial][np.argsort(rts)]
                )
                color_list.append(my_color)

        my_rasterPSTH(
        c.spikes.times,sel_spike_clusIDs,#c.spikes.clusters,
        ev_list,[1],
        pre_time=pre_times[i],post_time=post_times[i],smoothing=0.02,bin_size=0.01,
        include_raster=True,n_rasters=10,
        event_colors=color_list,reverse_raster=True,
        ax=ax[0,2*ai+i],ax1=ax[1,2*ai+i])



ax[0,0].set_title('confict')
ax[0,2].set_title('visual')
ax[0,4].set_title('coherent')

[ax[0,i].set_yticks([]) for i in range(1,6)]
[ax[0,i].set_ylabel('') for i in range(1,6)]

[ax[1,i*2].set_xlabel('Time after stim') for i in range(3)]
[ax[1,i*2+1].set_xlabel('Time after move') for i in range(3)]

off_axes(ax[0,6])
ax[1,6].scatter(np.ones(8),contrast_options,c=colors)
off_excepty(ax[1,6])
ax[1,6].set_ylabel('sgn(contrast)')
ax[1,6].yaxis.tick_right()
#    ax.plot(1,v,'o',color=my_color)
#     sel_choice = 2 

expname = '%(subject)s_%(expDate)s_%(probeID)s' % (rec)
fig.suptitle('%s_%s' % (expname,type))



# %%
# do the same trigger to collect the raster, and store the average for each neuron? 
# from Analysis.neural.utils.spike_dat import get_binned_rasters
# raster_kwargs = {
#     'pre_time':t_before,
#     'post_time':t_after, 
#     'bin_size':t_bin,
#     'smoothing':0,
#     'return_fr':False,
#     'baseline_subtract': False, 
#     }

# for ai,a_ind in enumerate(all_aud_pos):
#     req_a_pos = np.sign(contrast_options)*a_ind

#     for i,align_time in enumerate(align_time_types):
#         ev_list,color_list = [],[]
#         for v,a,choice,my_color in zip(contrast_options,req_a_pos,my_choices,colors):
#             trialID = np.where((visDiff==v) & (audPos==a) & (choices==choice))
#             is_current_trial = c.ev.trial_type_IDs==trialID[0]

#                 if np.sum(is_current_trial)>0:
#                     rts = c.ev.rt[is_current_trial]
#                     ev_list.append(
#     r = get_binned_rasters(c.spikes.times,
#                         c.spikes.clusters,
#                         c.clusters._av_IDs,
#                         ev[t_on_key],**raster_kwargs)



# %%
