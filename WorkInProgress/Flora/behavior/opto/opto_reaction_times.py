# %% 
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_github_paths
add_github_paths()
from Analysis.helpers.queryExp import load_data,Bunch
from opto_utils import select_opto_sessions

def get_rts_stim(ev,contrast = 40,laser_trials=False):
    # get RT for left vs right 
    # as well as no. of nogos? 
    try: 

        if laser_trials: 
            ev = Bunch({k:ev[k][ev.is_laserTrial] for k in ev.keys()})
        else: 
            ev = Bunch({k:ev[k][~ev.is_laserTrial] for k in ev.keys()})

        rt  = ev.timeline_choiceMoveOn - np.nanmin(np.concatenate([ev.timeline_audPeriodOn[:,np.newaxis],ev.timeline_visPeriodOn[:,np.newaxis]],axis=1),axis=1)

        # break things down to left vs right
        # left 

        rt_blank = rt[ev.is_blankTrial & ev.is_validTrial & ~np.isnan(ev.timeline_choiceMoveDir)]

        rt_aud_left = rt[ev.is_auditoryTrial & 
        ev.is_validTrial & 
        (ev.stim_audAzimuth == -60)
        ]

        rt_aud_right = rt[ev.is_auditoryTrial & 
        ev.is_validTrial & 
        (ev.stim_audAzimuth == 60)
        ]

        rt_vis_left = rt[ev.is_visualTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) & 
        (ev.stim_visAzimuth == -60)
        ]

        rt_vis_right = rt[ev.is_visualTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) & 
        (ev.stim_visAzimuth == 60)
        ]

        rt_coh_left = rt[ev.is_coherentTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) & 
        (ev.stim_audAzimuth == -60)
        ]


        rt_coh_right = rt[ev.is_coherentTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) & 
        (ev.stim_audAzimuth == 60)
        ]

        rt_conf_aud_left = rt[ev.is_conflictTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) &
        (ev.stim_audAzimuth == -60)
         ]

        rt_conf_aud_right = rt[ev.is_conflictTrial & 
        ev.is_validTrial & 
        ((ev.stim_visContrast*100).astype('int')==contrast) &
        (ev.stim_audAzimuth == 60)
         ]

    except AttributeError:
       rt_blank = None
       rt_aud_left,rt_aud_right,rt_vis_left,rt_vis_right = None,None,None,None
       rt_coh_left,rt_coh_right,rt_conf_aud_left,rt_conf_aud_right = None,None,None,None

    return rt_blank,rt_aud_left,rt_aud_right,rt_vis_left,rt_vis_right,rt_coh_left,rt_coh_right,rt_conf_aud_left,rt_conf_aud_right


def get_rts_choice(ev,laser_trials=False): 
    try: 

        if laser_trials: 
            ev = Bunch({k:ev[k][ev.is_laserTrial] for k in ev.keys()})
        else: 
            ev = Bunch({k:ev[k][~ev.is_laserTrial] for k in ev.keys()})

        rt  = ev.timeline_choiceMoveOn - np.nanmin(np.concatenate([ev.timeline_audPeriodOn[:,np.newaxis],ev.timeline_visPeriodOn[:,np.newaxis]],axis=1),axis=1)

        left = ev.is_validTrial & (ev.timeline_choiceMoveDir==1)
        right = ev.is_validTrial & (ev.timeline_choiceMoveDir==2)

        rt_left = rt[left]
        rt_right = rt[right]

    except: 
        rt_left,rt_right = None,None

    return rt_left, rt_right


def shuffle_and_cut(myarray,n):
    """
    in order to balance trial numbers, 
    we typically shuffle trials and take the first x 

    """
    np.random.seed(0)
    np.random.shuffle(myarray)

    return myarray[:n]

def get_relative_rts(selected_recordings,dropped = ['blank','conflict']):
    df = selected_recordings.drop(labels=['Subject','expDate','expNum','rigName','expDuration'],axis='columns')
    # fdrop further columns specified by dropped 
    for dropped_cols in dropped: 
        df = df.loc[:,~df.columns.str.contains(dropped_cols)]

    all_rts_wnogo = Bunch({colname:np.concatenate(colval.values) for colname,colval in df.iteritems()})
    # throw away nans
    all_rts = Bunch({k: all_rts_wnogo[k][~np.isnan(all_rts_wnogo[k])] for k in all_rts_wnogo.keys()})

    min_trial_number = np.min([all_rts[k].size for k in all_rts.keys()]) # this should be able to exclude some trial types

    # get equal number of trials
    all_rts_trimmed  = {k: shuffle_and_cut(all_rts[k],min_trial_number) for k in all_rts.keys()}
    all_rts_trimmed = pd.DataFrame.from_dict(all_rts_trimmed)

    tot = np.median(all_rts_trimmed.values)

    all_rel_rts = pd.DataFrame.from_dict({colname:np.array([np.median(colval.values)]) for colname,colval in all_rts_trimmed.iteritems()})

    return all_rts_wnogo,all_rel_rts

def get_sessions(contrast = 40, sess_laser_power = 30, sess_hemisphere = 'R', **kwargs):

    data_dict = {
        'events':{'_av_trials':'table'}
        }

    kwargs['expDef'] = 'multiSpace'
    recdat = load_data(data_name_dict = data_dict,**kwargs)
    recdat = select_opto_sessions(recdat)
    # from recdat drop recordings that are too short
    recdat = recdat[(recdat.laser_power==sess_laser_power) & 
    (recdat.stimulated_hemisphere == sess_hemisphere) & 
    (recdat.extractEvents=='1')
    ]

    out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration']]    
    out_dat = out_dat.reset_index(drop=True)
    # performance measures    
    
    b,a_l,a_r,v_l,v_r,coh_l,coh_r,conf_l,conf_r = zip(*[get_rts_stim(rec.events._av_trials,contrast=contrast,laser_trials=False) for _,rec in recdat.iterrows()])
    b_laser,a_l_laser,a_r_laser,v_l_laser,v_r_laser,coh_l_laser,coh_r_laser,conf_l_laser,conf_r_laser = zip(*[get_rts_stim(rec.events._av_trials,contrast=contrast,laser_trials=True) for _,rec in recdat.iterrows()])
    left_rt,right_rt = zip(*[get_rts_choice(rec.events._av_trials,laser_trials=False) for _,rec in recdat.iterrows()])
    left_rt_laser,right_rt_laser = zip(*[get_rts_choice(rec.events._av_trials,laser_trials=True) for _,rec in recdat.iterrows()])

    out_dat = out_dat.assign(
        blank_rt = b,
        aud_left_rt = a_l, 
        aud_right_rt = a_r, 
        vis_left_rt = v_l,
        vis_right_rt = v_r,
        coherent_left_rt = coh_l,
        coherent_right_rt = coh_r, 
        conflict_audLeft_rt = conf_l, 
        conflict_audRight_rt = conf_r,
        blank_rt_laser = b_laser,
        aud_left_rt_laser = a_l_laser, 
        aud_right_rt_laser = a_r_laser, 
        vis_left_rt_laser = v_l_laser,
        vis_right_rt_laser = v_r_laser,
        coherent_left_rt_laser = coh_l_laser,
        coherent_right_rt_laser = coh_r_laser, 
        conflict_audLeft_rt_laser = conf_l_laser, 
        conflict_audRight_rt_laser = conf_r_laser, 
        left_rt = left_rt,
        right_rt = right_rt,
        left_rt_laser = left_rt_laser,
        right_rt_laser = right_rt_laser
    )

    return out_dat 
# %%
mname = 'AV029'
power = 15
hemisphere = 'R'
contrast = 40
recordings = get_sessions(subject = [mname],contrast=contrast,sess_laser_power = power, sess_hemisphere = hemisphere)
rt_dists,rts = get_relative_rts(recordings,dropped = ['blank','conflict'])
# %%
import matplotlib.pyplot as plt
# plot things
non_laser_rts = rts.loc[:,~rts.columns.str.contains('laser')]
laser_rts = rts.loc[:,rts.columns.str.contains('laser')]

_,ax = plt.subplots(1,1,figsize = (10,5))

non_laser_color_list = ['magenta','magenta','blue','blue','green','green']
laser_color_list = ['plum','plum','lightblue','lightblue','lightgreen','lightgreen']

ct = 0
for colname,colval in non_laser_rts.iteritems():
    ax.bar(ct*2,colval.values,color = non_laser_color_list[ct])
    ax.bar(ct*2+1,laser_rts.loc[:,laser_rts.columns.str.contains(colname)].values[0],color = laser_color_list[ct])
    ct+=1
# %%
# plot cumulative distribtions of reaction times for left vs right choices 

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].hist(rt_dists.right_rt,bins=40,alpha=0.7,color='k')
ax[0].hist(rt_dists.right_rt_laser,bins=40,alpha=0.4,color='lime')
ax[0].legend(['noLED: %.2f' % (np.nanmedian(rt_dists.right_rt)),'LED: %.2f' % (np.nanmedian(rt_dists.right_rt_laser))])
ax[0].set_title('right choices')
ax[0].set_xlabel('reaction time (s)')

ax[1].hist(rt_dists.left_rt,bins=40,alpha=0.7,color='k')
ax[1].hist(rt_dists.left_rt_laser,bins=40,alpha=0.7,color='lime')
ax[1].legend(['noLED: %.2f' % (np.nanmedian(rt_dists.left_rt)),'LED: %.2f' % (np.nanmedian(rt_dists.left_rt_laser))])
ax[1].set_title('left choices')
ax[1].set_xlabel('reaction time (s)')
fig.suptitle('%s,%.0f mW, %s hemisphere inhibited' % (mname,power,hemisphere))

# %% nogos that we look at for each side in the various trial 

rt_keys = list(rt_dists.keys())
rt_keys = rt_keys[:12]
non_laser_color_list = ['magenta','magenta','blue','blue','green','green']
laser_color_list = ['plum','plum','lightblue','lightblue','lightgreen','lightgreen']

laser_keys = [s for s in rt_keys if "laser" in s]
non_laser_keys = [s for s in rt_keys if "laser" not in s]
fig,ax = plt.subplots(1,1,figsize = (10,5))
labels = []
for i,(nk,k) in enumerate(zip(non_laser_keys,laser_keys)):
    ax.bar(2*i,np.isnan(rt_dists[nk]).mean(),color = 'k')
    ax.bar(2*i+1,np.isnan(rt_dists[k]).mean(),color = 'lime')
    labels.append(nk)
    labels.append(k)

ax.set_xticks(np.arange(12),rotation=45)
ax.set_xticklabels(labels,rotation=60)
ax.set_ylabel('proportion of misses')
fig.suptitle('%s,%.0f mW, %s hemisphere inhibited' % (mname,power,hemisphere))

# %%
