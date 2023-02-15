# %%
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.utils.plotting import off_topspines,off_axes

from Analysis.neural.src.kernel_model import kernel_model
kernels = kernel_model(t_bin=0.005,smoothing=0.025)

nrn_list = [2,7,234,276]
#nrn_list = [50,140]
kernels.load_and_format_data(
    subject = 'FT009',
    expDate = '2021-01-20', 
    expDef = 'all',
    expNum = 7,
    probe = 'probe0',
    subselect_neurons=nrn_list,
    t_support_stim = [-0.05,0.6],
    t_support_movement =[-.6,0.4],
    rt_params = {'rt_min': None, 'rt_max': None},
    event_types = ['aud','vis','baseline','coherent-non-linearity'],
    contrasts = [1],
    spls = [.02,.1],
    vis_azimuths = [-90,0,90],
    aud_azimuths = [-90,0,90],
    digitise_cam = False,
    zscore_cam= 'mad' 
)

# perform fitting

kernels.fit(method='Ridge',ridge_alpha=1,tune_hyper_parameter=False,rank=10,rr_regulariser=0)
kernels.predict()
variance_explained = kernels.evaluate(kernel_selection = 'stimgroups',sig_metric = ['explained-variance','explained-variance-temporal'])

# %% 
# look at the VE over the trial
import matplotlib.pyplot as plt
n = 276

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
# look at the average predictions data and prediction. 
# literally do the equivalent of the AVmodel plot.
kernels.plot_prediction(nrnID=n,plot_stim = True, plot_move=False, sep_choice=False)


# %%
