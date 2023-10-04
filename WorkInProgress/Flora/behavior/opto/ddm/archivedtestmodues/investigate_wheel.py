# %%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.pyutils.plotting import off_axes,off_topspines
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV036']

#subject = ['AV038']
#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2021-05-02:2023-09-04',expDef='multiSpace',data_name_dict=data_dict)
recordings = recordings[recordings.extractEvents=='1']
# throw away sessions without any laser    

# %%
to_keep = [(rec.events._av_trials.is_conflictTrial.sum()>5) for _,rec in recordings.iterrows() ]
recordings = recordings[to_keep]
# %%
def sort_curr_condition(ev):
        passcond = (ev.is_visualTrial & ev.is_validTrial & 
                    ((ev.stim_visContrast*np.sign(ev.stim_visAzimuth)==-np.sort(np.unique(ev.stim_visContrast))[-2])|(ev.stim_visContrast*np.sign(ev.stim_visAzimuth)==np.sort(np.unique(ev.stim_visContrast))[-2]))
                     & (~np.isnan(ev.timeline_choiceMoveOn)))
        return passcond

to_keep = [((sort_curr_condition(rec.events._av_trials)).sum())>1 for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

# %%

def batch_rasters(rec,**wheel_kwargs): 
    # sort the indices
    print(rec.expFolder)
    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    idxs = np.where(sort_curr_condition(ev))[0]
    my_wheel = wheel_raster(ev,selected_trials=idxs, **wheel_kwargs)    
    r = my_wheel.rasters     
    return r,ev.rt[idxs],ev.rt_aud[idxs],ev.timeline_choiceMoveDir[idxs],my_wheel.tscale


t_bin = 0.005
ts = [-.2,1]
r,rts,rts_,choicedir,tscale = zip(*[batch_rasters(rec,align_type ='aud',t=ts,t_bin=t_bin) for _,rec in recordings.iterrows()])
tscale = tscale[0]
rasters = np.concatenate(r)
rts,rts_= np.concatenate(rts),np.concatenate(rts_)
choicedir = np.concatenate(choicedir)

# %%
fig,ax = plt.subplots(1,1,figsize=(8,8))
colors = ['blue','red']
[ax.plot(tscale,w,color=colors[int(c-1)],alpha=.3) for w,c in zip(rasters,choicedir)]
ax.axhline(20,color='k',linestyle='--')
ax.axhline(-20,color='k',linestyle='--')
ax.set_xlim([-0.05,.2])
ax.axvline(0,color='k',linestyle='--')

ax.set_xlabel('time (s) from stim onset')
ax.set_ylabel('wheel turn (deg)')

# %%
fig,ax = plt.subplots(1,1,figsize=(15,15))
colors = ['blue','red']
[ax.plot(tscale[1:],np.diff(w),color=colors[int(c-1)],alpha=.3) for w,c in zip(rasters,choicedir)]

ax.set_xlim([-.2,.2])
ax.set_xlabel('time (s) from choice onset')
ax.set_ylabel('wheel velocity (deg/%.3f s)' % t_bin)
ax.plot(0,0,'o',markersize=11,color='k')
# %%
# or is there
fig,ax = plt.subplots(1,1,figsize=(8,8))

rts_idx_style = (rts/t_bin)+(-ts[0]/t_bin)
ax.axvline((-ts[0]/t_bin),color='k')
ax.matshow(rasters[np.argsort(rts),:],aspect='auto',cmap='coolwarm_r',vmin=-.2,vmax=.2)
#[ax.plot([r,r],[i-1,i],color='k',linewidth=5) for i,r in enumerate(np.sort(rts_idx_style))]


# %%
fig,ax = plt.subplots(1,1,figsize=(10,8))
ax.hist(rts[choicedir==1],bins=100,color='b',alpha=.3)
ax.hist(rts[choicedir==2],bins=100,color='r',alpha=.3)
ax.set_xlim([0, .05])
# %%

# plot the mean wheel tracesl between certan rt bins

rt_bins = np.linspace(0,0.02,5)
colors_r = plt.cm.Reds(np.linspace(.5,.9,rt_bins.size-1))

colors_l = plt.cm.Blues(np.linspace(.5,.9,rt_bins.size-1))

fig,ax = plt.subplots(1,1,figsize=(10,8))
for i in range(rt_bins.size-1):
    ax.plot(tscale,np.mean(rasters[(choicedir==1) & (rts<=rt_bins[i+1]) & (rts<=rt_bins[i]),:],axis=0),color=colors_l[i])
    ax.plot(tscale,np.mean(rasters[(choicedir==2) & (rts<=rt_bins[i+1]) & (rts<=rt_bins[i]),:],axis=0),color=colors_r[i])


#fig.legend(['rt = %.2f-%.2f s' % (rt_bins[i],rt_bins[i+1]) for i in range(rt_bins.size-1)],loc='upper right',bbox_to_anchor=(1.1,1.1))

#ax.set_xlim([0,0.35])
ax.axhline(-20,color='k',linestyle='--')
ax.axhline(20,color='k',linestyle='--')
ax.set_xlabel('time from detected choice onset (s)')

# %%
