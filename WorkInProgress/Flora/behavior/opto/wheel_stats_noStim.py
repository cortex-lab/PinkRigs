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
from Analysis.pyutils.plotting import off_axes
data_dict = {
            'events':{'_av_trials':'all'}
                }

subject = ['AV046','AV047','AV041','AV044','AV036']
#subject = ['AV047']
#recordings = query_opto(subject=subject,expDate = 'all',expDef='multiSpace',data_dict=data_dict)
recordings = load_data(subject=subject,expDate='2022-04-25:2023-08-20',expDef='multiSpace',data_name_dict=data_dict)
recordings = recordings[recordings.extractEvents=='1']

#to_keep = [ ~(np.isnan(rec.events._av_trials.is_laserTrial).all()) for _,rec in recordings.iterrows()]

to_keep = [np.sum(rec.events._av_trials.is_laserTrial)>0 for _,rec in recordings.iterrows()]
recordings = recordings[to_keep]

# (rec.events._av_trials.is_noStimTrial.sum()>5) &


def batch_rasters(rec,return_shuffles=False): 
    # sort the indices
    print(rec.expFolder)

    ev,_,_,_,_= simplify_recdat(rec,reverse_opto=True)
    idxs = np.where(
        ev.is_noStimTrial &
        ev.is_laserTrial 
        #np.isnan(ev.timeline_choiceMoveOn)
        )[0]


    # plot an example trace
    laserMoveT = ev.timeline_firstMovePostLaserOn-ev.timeline_laserOn_rampStart
    rt_noStim = laserMoveT[idxs]

    DirMoveLaser = ev.timeline_firstMovePostLaserDir
    dir_noStim = DirMoveLaser[idxs]
    
    powers = ev.laser_power[idxs]

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
        align_type='laserOn',
        t = [-.1,3]
        )
    
    # reverse raster trace based on location # which is only advisable once you bseline subtracted....
    posarg = np.tile(np.sign(ev.laser_power_signed[idxs]),(my_wheel.rasters.shape[1],1)).T
    r = posarg * my_wheel.rasters

    expFolders = [rec.expFolder]*powers.size


    return r,rt_noStim,powers,shuffle_rts,dir_noStim,expFolders

r,td,p,td_shuff,dirs,expFolders = zip(*[batch_rasters(rec,return_shuffles=False) for _,rec in recordings.iterrows() ])
rasters = np.concatenate(r)
td = np.concatenate(td)
p= np.concatenate(p)
expFolders = np.concatenate(expFolders)

if td_shuff[0] is not None:
    td_shuff = np.concatenate(td_shuff,axis=1)
dirs = np.concatenate(dirs)

# %%

fig,ax = plt.subplots(1,1)

[ax.hist(x,bins=200,color='k',alpha=.3) for x in td_shuff]
ax.hist(td,bins=200,color='red',alpha=.3)

# %%
_,ax = plt.subplots(1,1,figsize=(6,6)) 
unique_powers = np.array([5,10,17]) # np.unique(p[p>0])
colors = plt.cm.viridis(np.linspace(0.2,.8,unique_powers.size))
for idx,i in enumerate(unique_powers):
    ax.hist(td[p==i],bins=100,cumulative=False,density=False,stacked=False,histtype='bar',alpha=0.6,color=colors[idx],lw=10) 


# %%
fig,ax = plt.subplots(1,1)
sel_idx = np.argsort(td)
ax.imshow((rasters[sel_idx,:400]),aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
#ax.imshow((rasters[630:830,:]),aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
#ax.matshow(rasters[np.argsort(rasters[:,300:].mean(axis=1)),:],aspect='auto',vmin=-.4,vmax=.4,cmap='coolwarm_r')
#off_axes(ax)
ax.axvline(100,color='lime')
#ax.plot(100,0,marker='v',markersize=30,color='lime')
#ax.plot(300,0,marker='v',markersize=30,color='grey')
#ax.vlines(0,1,100,'k',lw=6)
#ax.hlines(rasters.shape[0]*0.99,400,499,'k',lw=6)
#plt.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\optoWheel_noStim.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#ax.axvline(100,color='r')
#ax.set_title('%s,inactivated_side=%s,%.0fmW,align:laser,sort:audOn-laserOn' % (subject)) 
# %%
