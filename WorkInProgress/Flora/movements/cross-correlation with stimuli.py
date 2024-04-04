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
recordings = load_for_movement_correlation(dataset='naive')



# %
cs = ['vis','aud']
vis_d = []
aud_d = []
for _,rec in recordings.iterrows():        

    events,spikes,_,_,cam = simplify_recdat(rec,cam_hierarchy=['frontCam','eyeCam','sideCam'])

    if hasattr(cam,'ROIMotionEnergy') & hasattr(cam,'times'):
        x,y,upfactor = cam.times,cam.ROIMotionEnergy,100
        interp_func = scipy.interpolate.interp1d(x,y,kind='linear')
        times = np.linspace(np.min(x),np.max(x),x.size*upfactor)
        camtrace = interp_func(times)

        print('{subject}_{expDate}_{expNum}'.format(**rec))

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
# %%
plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':28})
from Analysis.pyutils.plotting import off_topspines
fig,ax = plt.subplots(1,2,figsize=(10,3),sharey=True,sharex=True)
for i in range(len(vis_d)):
    ax[0].plot(tvals,vis_d[i,:],'b',alpha=0.3,lw=2)
    ax[1].plot(tvals,aud_d[i,:],'m',alpha=0.3,lw=2)

ax[0].plot(tvals,vis_d.mean(axis=0),'b',alpha=0.8,lw=6)
ax[1].plot(tvals,aud_d.mean(axis=0),'m',alpha=0.8,lw=6)
ax[0].set_xticks([-0.8,0,0.8])
off_topspines(ax[0])
off_topspines(ax[1])

fig.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\movestim_crossCorr_naive.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=200)

# %%
