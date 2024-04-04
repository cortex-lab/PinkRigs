# %%




import sys
import sklearn
import scipy 
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import simplify_recdat
from Analysis.neural.utils.spike_dat import cross_correlation
from Analysis.pyutils.ev_dat import digitize_events
from Analysis.pyutils.plotting import off_topspines




from loaders import load_for_movement_correlation
recordings = load_for_movement_correlation(dataset='postactive',recompute_data_selection=False)
# %%
from Analysis.pyutils.video_dat import get_move_raster,plot_triggerred_data


def get_significant_LRdiff(rec,to_plot=False):
    """
    function that is used to test whether there  is significant differene in the evoke LR movement following auditory stimulus
    """    
    events = rec.events._av_trials
    cam = rec.camera

    timings = {
        'pre_time':.15,
        'post_time':0.45,
        'bin_size': .005
    }
    if 'is_validTrial' not in list(events.keys()):
        events.is_validTrial = np.ones(events.is_auditoryTrial.size).astype('bool')

    is_selected  = events.is_validTrial & (events.stim_audAmplitude>0) 

    left_vs_right = np.sign(events.stim_audAzimuth)

    azimuths = np.unique(left_vs_right)
    azimuths = azimuths[~np.isnan(azimuths) & (azimuths!=0)]
    #azimuths = np.array([-90,-60,-30,0,30,60,90])
    azi_colors = plt.cm.coolwarm(np.linspace(0,1,azimuths.size))



    dat_points={}
    for azi in azimuths: 
        is_called_trials = is_selected & (left_vs_right==azi)
        # sort by reaction time 
        on_times = events.timeline_audPeriodOn[is_called_trials & ~np.isnan(events.timeline_audPeriodOn)]

        raster,bins,idx = get_move_raster(
            on_times,cam.times,cam.ROIMotionEnergy,
            sortAmp=False,baseline_subtract=True,
            ax=None,to_plot=False,**timings
            )
        
        dat_points[azi] = raster[:,bins>0].mean(axis=1)

    mintrial = min([dat_points[k].size for k in dat_points.keys()])

    resps_per_azimuth = np.concatenate([dat_points[k][:mintrial][np.newaxis,:] for k in dat_points.keys()])



    LRdiff_actual = np.diff(resps_per_azimuth.mean(axis=1))
    LRmean = resps_per_azimuth.mean(axis=1)

    shuffle_no = 1000
    LRdiff_shuffled = np.zeros(shuffle_no)
    # can shuffle and reassign
    for s in range(shuffle_no):
        np.random.seed(s)
        np.random.permutation(np.ravel(resps_per_azimuth))
        LRdiff_shuffled[s] = np.diff((np.random.permutation(np.ravel(resps_per_azimuth))).reshape(2,mintrial).mean(axis=1))

    p_value = (np.abs(LRdiff_shuffled)>np.abs(LRdiff_actual)).sum()/shuffle_no

    if to_plot:
        fig,ax = plt.subplots(1,1,figsize=(4,4))
        ax.hist(LRdiff_shuffled,bins=100)
        ax.axvline(LRdiff_actual,color='r')

    return p_value,LRmean
# %%
    
p_values,LRdiffs = zip(*[get_significant_LRdiff(rec,to_plot=False) for _,rec in recordings.iterrows()])

p_values = np.array(p_values)
print('LR significance in %.2f of recordings' % (p_values<0.05).mean())
# %%

for i in range(len(LRdiffs)):
    cd = LRdiffs[i]
    cd = cd - min(cd)
    plt.plot(cd,'k')

# %%
# plot the ones that appear significant!

plot_triggerred_data(recordings=recordings[p_values<0.05])

# %%
