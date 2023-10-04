# %%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np
from Analysis.pyutils.wheel_dat import wheel_raster


from opto_utils import query_opto
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.pyutils.plotting import off_axes,off_topspines, off_excepty
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV036','AV038']
#subject = ['AV047']
#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2023-08-07:2023-09-03',expDef='multiSpace',data_name_dict=data_dict)
# %%
recordings = recordings[recordings.extractEvents=='1']

#to_keep = [ ~(np.isnan(rec.events._av_trials.is_laserTrial).all()) for _,rec in recordings.iterrows()]

to_keep = [np.sum(rec.events._av_trials.is_laserTrial)>0 for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

# (rec.events._av_trials.is_noStimTrial.sum()>5) &


def batch_rasters(rec,return_shuffles=False): 
    # sort the indices
    print(rec.expFolder)

    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=False)
    idxs = np.where(
        ev.is_noStimTrial  & ~ev.block_isMovedAtLaser      #np.isnan(ev.timeline_choiceMoveOn)
        )[0]

    curr_subject = np.array([rec.subject for x in range(idxs.size)])
    

    #  not quite good imho
    laserMoveT = ev.block_firstMovePostLaserOn-ev.block_laserStartTimes
    rt_noStim = laserMoveT[idxs]

    DirMoveLaser = ev.block_firstMovePostLaserDir
    dir_noStim = DirMoveLaser[idxs]
    
    powers = ev.laser_power_signed[idxs]

    if return_shuffles: 
        # conditional resampling for each of these 
        # means that for the same parameters of ITI # actually I don't need the actual ITI, just need to sample from the same distribution. 
        n = ev.is_noStimTrial.size
        n_shuffles = 1000
        shuffle_rts = []
        for shuffle in range(n_shuffles):
            iti = [np.min([0.5+np.random.exponential(scale=.25),1.5]) for i in range(n)]
            quiescent = [0.25+(-.15*np.random.uniform()) for i in range(n)]
            mts = [ev.timeline_allMoveOn[i]-ev.block_trialOn[i]+iti[i] for i in range(n)] 
            hold_durs_after_iti = [np.insert(np.diff(mt[mt>0]),0,mt[mt>0][0]) if (mt[mt>0].size>0) else np.empty(1)*np.nan for mt in mts]
            fake_rts = np.array([(hd[hd>quiescent[i]]-quiescent[i])[0] if  (hd>quiescent[i]).sum()>0 else np.nan for i,hd in enumerate(hold_durs_after_iti)])
            shuffle_rts.append(fake_rts[idx][np.newaxis,:])
        shuffle_rts = np.concatenate(shuffle_rts,axis=0)
    else:
        shuffle_rts = None

    my_wheel = wheel_raster(
        ev,
        selected_trials=idxs, 
        align_type='blockLaserTime',
        t = [-.1,2.0],t_bin=0.001
        )
    
    # reverse raster trace based on location # which is only advisable once you bseline subtracted....
    # posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    # r = posarg * my_wheel.rasters
    r = my_wheel.rasters
    expFolders = [rec.expFolder]*powers.size

    is_on = ev.is_laserTrial[idxs]

    return r,rt_noStim,is_on,powers,shuffle_rts,dir_noStim,expFolders,my_wheel.tscale,curr_subject

r,td,is_laserOn,p,td_shuff,dirs,expFolders,tscale,subject_per_trial = zip(*[batch_rasters(rec,return_shuffles=False) for _,rec in recordings.iterrows()])
rasters = np.concatenate(r)
td = np.concatenate(td)
is_laserOn = np.concatenate(is_laserOn)
p= np.concatenate(p)
expFolders = np.concatenate(expFolders)
tscale = tscale[0]
subject_per_trial = np.concatenate(subject_per_trial)

if td_shuff[0] is not None:
    td_shuff = np.concatenate(td_shuff,axis=1)
dirs = np.concatenate(dirs)

# %%

# fig,ax = plt.subplots(1,1)
# plot for shuffling

# [ax.hist(x,bins=200,color='k',alpha=.3) for x in td_shuff]
# ax.hist(td,bins=200,color='red',alpha=.3)

# %%
_,ax = plt.subplots(1,1,figsize=(6,6)) 
unique_powers = np.array([0,2,5,10,17]) # np.unique(p[p>0])
colors = plt.cm.viridis(np.linspace(0.2,.8,unique_powers.size))
for idx,i in enumerate(unique_powers):
    ax.hist(td[p==i],bins=100,cumulative=False,density=False,stacked=False,histtype='bar',alpha=0.6,color=colors[idx],lw=10) 





# %%
fig,ax = plt.subplots(1,1)
sel_idx = np.argsort(td)


#selected_l = (p==10)
selected_l = np.isnan(p)

rasters_sel = rasters[selected_l,:]
td_selected = td[selected_l]
n_subsample =np.random.permutation(np.arange(0,np.sum(selected_l),1))

n_length = 1000
raster_sel_sorted = rasters_sel[n_subsample[:n_length],:]
ax.matshow((raster_sel_sorted[np.argsort(td_selected[n_subsample[:n_length]]),:]),aspect='auto',vmin=-10,vmax=10,cmap='coolwarm_r')
off_axes(ax)
#ax.imshow((rasters[630:830,:]),aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
#ax.matshow(rasters[np.argsort(rasters[:,300:].mean(axis=1)),:],aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
#off_axes(ax)
# ax.axvline(100,color='k')
# ax.hlines(300,400,500,color='k',lw=6)
# ax.vlines(0,0,100,color='k',lw=6)

#plt.colorbar()
#%%
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Calibri Light'})

_,ax = plt.subplots(1,1,figsize=(13,8))

t_thr=1.8

ax.hist(td[np.isnan(p) & (td<t_thr)],bins=125,cumulative=False,density=True,stacked=False,histtype='bar',alpha=0.6,color='k',lw=10)

#ax.hist(td[(np.abs(p)>0) & (td<t_thr)],bins=125,cumulative=False,density=True,stacked=False,histtype='bar',alpha=0.6,color='lime',lw=10)

ax.hist(td[(np.abs(p)==0) & (td<t_thr)],bins=125,cumulative=False,density=True,stacked=False,histtype='bar',alpha=0.6,color='orange',lw=10)


off_topspines(ax)
ax.set_xlabel('RT from end of quiescent period (s)')
ax.set_ylabel('rel. density')
# ax.vlines(np.median(td[np.isnan(p) & (td<t_thr)]),3,4.3,color='k',lw=4)

# ax.vlines(np.median(td[(np.abs(p)>0) & (td<t_thr)]),3,4.3,color='lime',lw=4)

#ax.hist(td[(np.abs(p)==10) & (td<1.5)],bins=75,cumulative=True,density=True,stacked=False,histtype='bar',alpha=0.6,color='red',lw=10)
#ax.hist(td[(np.abs(p)==17) & (td<1.5)],bins=75,cumulative=True,density=True,stacked=False,histtype='bar',alpha=0.6,color='brown',lw=10)

#ax.plot(100,0,marker='v',markersize=30,color='lime')
#ax.plot(300,0,marker='v',markersize=30,color='grey')
#ax.vlines(0,1,100,'k',lw=6)
#ax.hlines(rasters.shape[0]*0.99,400,499,'k',lw=6)
#plt.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\optoWheel_noStim.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#ax.axvline(100,color='r')
#ax.set_title('%s,inactivated_side=%s,%.0fmW,align:laser,sort:audOn-laserOn' % (subject)) 
# %%
# plot the actual wheel traces

# %%
unique_powers = np.array([-17,-10,-5-2,2,5,10,17]) # np.unique(p[p>0])
colors = plt.cm.viridis(np.linspace(0.2,.8,unique_powers.size))
for idx,i in enumerate(unique_powers):
    plt.plot(tscale,np.nanmean(rasters[p==i],axis=0),color=colors[idx])
# how the trace scales with power (or not)
# %%
# 

cond_l = (dirs==1) & (p==0) #& np.isnan(p)
cond_r = (dirs==2) & (p==0)#& np.isnan(p)


rasters_l = rasters[cond_l,:]
rasters_r = rasters[cond_r,:]

colors_l = plt.cm.Blues_r(np.linspace(0.1,1,rasters_l.shape[0]))
colors_r = plt.cm.Reds_r(np.linspace(0.1,1,rasters_r.shape[0]))
_,ax= plt.subplots(1,1,figsize=(8,10))
[ax.plot(tscale,-rasters_l[ir,:],color='blue',alpha=.2) for i,ir in enumerate(np.argsort(td[cond_l]))] # nan is the true 

[ax.plot(tscale,-rasters_r[ir,:],color='red',alpha=.2) for i,ir in enumerate(np.argsort(td[cond_r]))] # nan is the true 

off_topspines(ax)
ax.axvline(0,color='k',linestyle='--')
ax.axhline(20,color='k',linestyle='--')
ax.axhline(-20,color='k',linestyle='--')
ax.set_ylim([-70,70])
ax.set_xlabel('time from quiescence end (s)')
ax.set_ylabel('wheel turn (deg)')

# %%
# have to be per subject 
fig,ax=plt.subplots(1,1,figsize=(5,7))
sel_s = np.unique(subject_per_trial)
for s in sel_s:
   # pR=np.nanmean(dirs[[(p==10) & (subject_per_trial==s)]]-1)
   # pL=np.nanmean(dirs[[(p==-10) & (subject_per_trial==s)]]-1)
    pR = np.mean(dirs[[(p==10) & (subject_per_trial==s)]]==2)
    pL = np.mean(dirs[[(p==-10) & (subject_per_trial==s)]]==2)

    ax.plot([1,2],[pL,pR],color='k')
    ax.plot([1,2],[pL,pR],'.',color='k',markersize=20)

pR=np.nanmean(dirs[[(p==10) & (subject_per_trial=='AV046')]]-1)
ax.plot(2,pR,'.',color='k',markersize=20)
ax.set_ylim([0,1])
ax.axhline(0.5,color='k',linestyle='--')
off_excepty(ax)

# %%
