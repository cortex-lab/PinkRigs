# %%
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.utils.plotting import off_topspines,off_axes

from Analysis.neural.src.kernel_model import kernel_model
kernels = kernel_model(t_bin=0.005,smoothing=0.025)

from kernel_params import get_params
dat_params,fit_params,eval_params = get_params()
nrn_list = [571]
#nrn_list = [50,140]
kernels.load_and_format_data(
    subject = 'FT009',
    expDate = '2021-01-20', 
    expDef = 'all',
    expNum = 7,
    probe = 'probe0',
    subselect_neurons=None,
    **dat_params
)

# perform fitting

kernels.fit(**fit_params)
kernels.predict()
variance_explained = kernels.evaluate(**eval_params)

# %% 
# look at the VE over the trial
import matplotlib.pyplot as plt
n = 208

plt.rcParams.update({'font.family':'Verdana'})
plt.rcParams.update({'font.size':16})
plt.rcParams['figure.dpi'] = 300
color_dict = {
    'aud': 'magenta',
    'vis': 'blue',
    'motionEnergy': 'black',
    'non-linearity':'green'
}

ve_n = variance_explained[
    (variance_explained.cv_number==0) & 
    (variance_explained.clusID==n)
    ]
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
kernels.plot_prediction(
    nrnID=n,
    plot_stim = True, 
    plot_move=False, 
    sep_choice=False,
    plotted_vis_azimuth = np.array([-1000,-90,-60,-30,0,30,60,90]),
    plotted_aud_azimuth = np.array([-1000,-90,-60,-30,0,30,60,90]),
    plot_train =True,
    plot_test = True,
    plot_pred_train = False,
    plot_pred_test = True,
    )

# %%
# actually test and show all the kernels
kernels.fit_evaluate(get_prediciton=True,**fit_params)
kernels.plot_kernels(n)

# %%
