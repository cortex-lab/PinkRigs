# %%
import sys
import sklearn
import scipy 
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat
from Analysis.neural.utils.spike_dat import cross_correlation
from Analysis.pyutils.ev_dat import digitize_events

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
dat_type = 'naive-allen'
dat_keys = get_data_bunch(dat_type)

cs = ['vis','aud']

vis_d = []
aud_d = []
for _,rec_info in dat_keys.iterrows():
#  %
    if 'probe0' in rec_info.probe: 
        ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'_av_IDs'}
        other_ = {'events': {'_av_trials': 'table'},'frontCam':{'camera':['times','ROIMotionEnergy']}}

        recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_, **rec_info)
        if recordings.shape[0] == 1:            
            recordings =  recordings.iloc[0]
        else:
            print('recordings are ambiguously defined. Please recall.')

        events,spikes,_,_,cam = simplify_recdat(recordings,probe='probe')

        if hasattr(cam,'ROIMotionEnergy') & hasattr(cam,'times'):
            x,y,upfactor = cam.times,cam.ROIMotionEnergy,100
            interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
            times = np.linspace(np.min(x),np.max(x),x.size*upfactor)
            camtrace = interp_func(times)

            print('%s_%s_%.0f_%s' % tuple(rec_info))

            for which in cs:
                if 'vis' in which:
                    on = events.timeline_visPeriodOn[events.is_visualTrial]
                    off = events.timeline_visPeriodOff[events.is_visualTrial]
                elif 'aud' in which:
                    on = events.timeline_audPeriodOn[events.is_auditoryTrial]
                    off = events.timeline_audPeriodOff[events.is_auditoryTrial]

                off = on + 0.01
                ev_trace = digitize_events(on,
                                off,
                                times)
                        
                t_shift = 0.05
                n_shifts = int(t_shift/np.diff(times).mean())
                n_corrs = 20
                shifts = np.arange(-n_shifts*n_corrs,n_shifts*n_corrs,n_shifts)
                shifted_evs = np.concatenate([np.roll(ev_trace,s)[:,np.newaxis] for s in shifts],axis=1)

                cvals = cross_correlation(camtrace,shifted_evs)
                tvals = np.arange(-t_shift*n_corrs,t_shift*n_corrs,t_shift)
                if 'vis' in which:
                    vis_d.append(cvals[np.newaxis,:])
                elif 'aud' in which:
                    aud_d.append(cvals[np.newaxis,:])

        #ax.plot(tvals,cvals)
vis_d = np.array(np.concatenate(vis_d))
aud_d = np.array(np.concatenate(aud_d))

fig,ax = plt.subplots(1,1)
for i in range(len(vis_d)):
    ax.plot(tvals,vis_d[i,:],'b',alpha=0.5)
    ax.plot(tvals,aud_d[i,:],'m',alpha=0.5)

ax.plot(tvals,vis_d.mean(axis=0),'b',alpha=1,lw=8)
ax.plot(tvals,aud_d.mean(axis=0),'m',alpha=1,lw=8)

# %%
