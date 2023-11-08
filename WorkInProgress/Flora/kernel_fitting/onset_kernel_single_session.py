# %% 
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_topspines,off_axes

from Analysis.neural.src.kernel_model import kernel_model
kernels = kernel_model(t_bin=0.005,smoothing=0.025)

nrn_list = [22,25,50,71,80,207,34,156,325]


#nrn_list = [50,140]
kernels.load_and_format_data(
    subject = 'AV025',
    expDate = '2022-11-10', 
    expDef = 'all',
    expNum = 2,
    probe = 'probe0',
    subselect_neurons=None,
    t_support_stim = [-0.05,0.6],
    t_support_movement =[-.6,0.4],
    rt_params = {'rt_min': None, 'rt_max': None},
    event_types = ['aud','vis','baseline','motionEnergy'],
    contrasts = 'all',
    spls = 'all',
    #vis_azimuths = [-90,-60,-30,0,30,60,90],
    #aud_azimuths = [-90,-60,-30,0,30,60,90],
    vis_azimuths = [-60,0,60],
    aud_azimuths = [-60,0,60],
    digitise_cam = False,
    zscore_cam= 'mad'
)


# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Verdana'})
plt.rcParams.update({'font.size':20})
plt.rcParams['figure.dpi'] = 300

kernels.fit(method='Ridge',ridge_alpha=1,tune_hyper_parameter=False,rank=10,rr_regulariser=0)
variance_explained = kernels.evaluate(kernel_selection = 'stimgroups',sig_metric = ['explained-variance','explained-variance-temporal'])
kernels.fit_evaluate(get_prediciton=True,method='Ridge',ridge_alpha=1,tune_hyper_parameter=False,rank=10,rr_regulariser=0)

# %%
import matplotlib.pyplot as plt
n = 18
plt.rcParams.update({'font.family':'Verdana'})
plt.rcParams.update({'font.size':16})
plt.rcParams['figure.dpi'] = 300
color_dict = {
    'aud': 'magenta',
    'vis': 'blue',
    'motionEnergy': 'black',
    'non-linearity':'green'
}

ve_n = variance_explained[(variance_explained.cv_number==0) & (variance_explained.clusID==n)]
fig,ax = plt.subplots(1,1,figsize=(7,4))
stim_bin_range = np.arange(-0.05,0.6,kernels.t_bin)
[ax.plot(stim_bin_range,r.VE_trial,color=color_dict[r.event],lw=6) for _,r in ve_n.iterrows() if 'baseline' not in r.event]
# prepare this plot properly
first_stim_onset = np.min(np.array([kernels.events.timeline_audPeriodOn,kernels.events.timeline_visPeriodOn]),axis=0)
av_delay_on = np.nanmean((first_stim_onset-kernels.events.block_stimOn))
stim_off = np.max(np.array([kernels.events.timeline_audPeriodOff,kernels.events.timeline_visPeriodOff]),axis=0)
av_delay_off = np.nanmean((stim_off-kernels.events.block_stimOn))


ax.fill_betweenx(np.array([-.05,0.45]),av_delay_on,x2=av_delay_off,alpha=.1,color='r')
off_topspines(ax)
ax.legend([r.event for _,r in ve_n.iterrows() if 'baseline' not in r.event])
ax.set_xlabel('time during trial')
ax.set_ylabel('VE,test')
# %%
v_azimuths = [-1000,90]
v_contrasts = [0,.4]
a_azimuths = [90,-1000]
a_spls = [.25,0]
raster_kwargs = {'t_before': 0.05,'t_after': 0.4,'sort_idx': None}
kernels.plot_prediction_rasters(n,visual_azimuth=v_azimuths,auditory_azimuth=a_azimuths,contrast=v_contrasts,spl=a_spls) 
kernels.plot_kernels(n)

#kernels.plot_prediction(n,plot_stim=True,plot_move=False,sep_choice=False,plot_train=True,plot_pred=True,plot_test=False)
#full_feature_matrix = kernels.feature_matrix.copy()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

#n = 219
nrn = np.where(np.array(kernels.clusIDs)==n)[0][0]
#ax[0].plot(kernels.prediction[nrn,:],kernels.R[nrn,:],'.')
# %
on_time = kernels.events.timeline_audPeriodOn[~np.isnan(kernels.events.timeline_audPeriodOn) & (kernels.events.stim_audAzimuth==90)]
r = kernels.get_raster(on_time,t_before=.05,t_after =.5,spike_type='data')
p = kernels.get_raster(on_time,t_before=.05,t_after =.5,spike_type='pred')

x = np.ravel(p.raster[nrn,:,:])
y = np.ravel(r.raster[nrn,:,:])

# look 
fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.histplot(x=x,y=y,bins=(30,30),ax=ax,color='red')
ax.plot(np.array([0,1]),np.array([0,1]),'k--')
ax.set_xlabel('prediction')
ax.set_ylabel('actual')

# %%
full_feature_matrix = kernels.feature_matrix.copy()
to_omit = ['aud_kernel_spl_0.10_azimuth_90','vis_kernel_contrast_1.00_azimuth_90',
'vis_kernel_contrast_1.00_azimuth_-90','aud_kernel_spl_0.10_azimuth_-90']
feature_matrix_omitted = full_feature_matrix.copy()
if to_omit: 
    for k in to_omit:    
        feature_matrix_omitted[:, kernels.feature_column_dict[k]] = 0
        kernels.feature_matrix = feature_matrix_omitted
else:
        kernels.feature_matrix = full_feature_matrix   

kernels.fit_evaluate(get_prediciton=True,method='Ridge',ridge_alpha=1,tune_hyper_parameter=False,rank=10,rr_regulariser=0)
kernels.plot_kernels(n)
kernels.plot_prediction(n,plot_stim=True,sep_choice=False,plot_move=False)

# %% 
