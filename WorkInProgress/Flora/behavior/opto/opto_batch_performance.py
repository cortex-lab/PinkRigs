# %%
#  this sript should allow plotting of 
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_PinkRigs_to_path
add_PinkRigs_to_path()
from Admin.csv_queryExp import load_data,Bunch
from opto_utils import select_opto_sessions


# total performance
# performance on opto+1 trials 
# rel. reaction time per condition
# rel. nogo per condition 
# always show the trial number

def get_choices_per_contrast(ev,laser_trials = False,subsequent_trial=False):
    try: 
        if laser_trials: 
            if subsequent_trial: 
                ev.is_laserTrial_shift1 = np.roll(ev.is_laserTrial,1)
                # basically we use a trick to take only trials that are not laser trials and follow the laser trial
                ev.is_laserTrial = (ev.is_laserTrial_shift1.astype('int')-2*ev.is_laserTrial.astype('int'))==1

            ev = Bunch({k:ev[k][ev.is_laserTrial] for k in ev.keys()})
        else:
            if subsequent_trial: 
                # then we exclude both the laser and the laser+1 trials from the analysis
                ev.is_laserTrial_shift1 = np.roll(ev.is_laserTrial,1)
                ev.is_laserTrial = ev.is_laserTrial_shift1 + ev.is_laserTrial

                 
            ev = Bunch({k:ev[k][~ev.is_laserTrial] for k in ev.keys()})

        n_valid_go = np.sum(ev.is_validTrial[~np.isnan(ev.timeline_choiceMoveDir)])
        # visual trial performance
        contrasts_sided = np.sign(ev.stim_visAzimuth)*ev.stim_visContrast
        contrasts_sided[np.isnan(contrasts_sided)] = 0 

        unique_contrasts = np.sort(np.unique(contrasts_sided))
        #unique_contrasts = unique_contrasts[np.abs(unique_contrasts)>0]

        choices_per_contrast_vis = [ev.timeline_choiceMoveDir[ev.is_validTrial & 
                        (ev.is_visualTrial + ev.is_blankTrial)  & 
                        ~np.isnan(ev.timeline_choiceMoveDir) & 
                        (contrasts_sided==c)] for c in unique_contrasts]

        choices_per_contrast_coh = [ev.timeline_choiceMoveDir[ev.is_validTrial & 
                        (ev.is_coherentTrial + ev.is_blankTrial)  & 
                        ~np.isnan(ev.timeline_choiceMoveDir) & 
                        (contrasts_sided==c)] for c in unique_contrasts]

        unique_aud_azimuths = np.unique(ev.stim_audAzimuth.astype('int'))

        choices_per_azimuth_aud = [ev.timeline_choiceMoveDir[ev.is_validTrial & 
                (ev.is_auditoryTrial + ev.is_blankTrial)  & 
                ~np.isnan(ev.timeline_choiceMoveDir) & 
                (ev.stim_audAzimuth==c)] for c in unique_aud_azimuths]

    except AttributeError:
        n_valid_go,unique_contrasts,unique_aud_azimuths,choices_per_contrast_vis,choices_per_azimuth_aud,choices_per_contrast_coh = None,None,None,None,None,None
 

    return n_valid_go,unique_contrasts,unique_aud_azimuths,choices_per_contrast_vis,choices_per_azimuth_aud,choices_per_contrast_coh


def get_performance_from_set_of_recordings(sel_recordings):
    all_choices_vis = { '%.2f' % c : 
        np.concatenate([rec.vis_choices[i] for _,rec in sel_recordings.iterrows()])
        for i,c in enumerate(sel_recordings.iloc[0].contrast_set)
    }

    p_r_vis= {k:(all_choices_vis[k]-1).mean() for k in all_choices_vis.keys()}

    all_choices_coh = { '%.2f' % c : 
        np.concatenate([rec.coherent_choices[i] for _,rec in sel_recordings.iterrows()])
        for i,c in enumerate(sel_recordings.iloc[0].contrast_set)
    }
    p_r_coh= {k:(all_choices_coh[k]-1).mean() for k in all_choices_vis.keys()}

    all_choices_aud =  {'%.2f' % c : 
        np.concatenate([rec.aud_choices[i] for _,rec in sel_recordings.iterrows()])
        for i,c in enumerate(sel_recordings.iloc[0].audazi_set)
    }

    p_r_aud= {k:(all_choices_aud[k]-1).mean() for k in all_choices_aud.keys()}

    return pd.DataFrame(p_r_vis,index=[0]),pd.DataFrame(p_r_aud,index=[0]),pd.DataFrame(p_r_coh,index=[0])



def get_sessions(laser_trials = False, sess_laser_power = 30, sess_hemisphere = 'R', **kwargs):

    data_dict = {
        'events':{'_av_trials':'table'}
        }

    kwargs['expDef'] = 'multiSpace'
    recdat = load_data(data_name_dict = data_dict,**kwargs)
    # from recdat drop recordings that are too short
    recdat = select_opto_sessions(recdat)
    # select sessions of the requested parameters
    recdat = recdat[(recdat.laser_power==sess_laser_power) & (recdat.stimulated_hemisphere == sess_hemisphere)]
    out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration']]    
    out_dat = out_dat.reset_index(drop=True)
    # performance measures    
    
    go,c,a,pr_vis,pr_aud,pr_coh = zip(*[get_choices_per_contrast(rec.events._av_trials,laser_trials=laser_trials,subsequent_trial=False) for _,rec in recdat.iterrows()])

    out_dat = out_dat.assign(
        n_go = go,
        contrast_set = c,
        audazi_set = a,
        vis_choices = pr_vis, 
        aud_choices = pr_aud,
        coherent_choices = pr_coh

    )
    return out_dat

def get_laser_sessions(**kwargs):
    laser_trials_rec = get_sessions(laser_trials=True,**kwargs)
    non_laser_trials_rec = get_sessions(laser_trials=False,**kwargs)
    valid_laser_recordings  = laser_trials_rec[(laser_trials_rec.n_go>50)]
    valid_no_laser_recordings = non_laser_trials_rec[(laser_trials_rec.n_go>50)] # idea being that it is the same order, whether you call laser trials or not.    

    # ideally here we should perform some concatenation
    return valid_no_laser_recordings,valid_laser_recordings


# %%
# code specific for getting opto experiments 
import matplotlib.pyplot as plt 
mname = 'AV031'
power = 15
inhibited_hemisphere = 'L'

no_laser_df,laser_df = get_laser_sessions(subject=mname, expDate = 'all', sess_laser_power = power, sess_hemisphere = inhibited_hemisphere)

# %%
plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':24})
valid_subjects = np.unique(laser_df.Subject)

all_v_laser = pd.DataFrame()
all_a_laser = pd.DataFrame()
all_c_laser = pd.DataFrame()

all_v_no_laser = pd.DataFrame()
all_a_no_laser = pd.DataFrame()
all_c_no_laser = pd.DataFrame()


for mouse in valid_subjects:
    # LASER TRIALS 
    sel_laser_recordings= laser_df[laser_df.Subject==mouse]
    v,a,c = get_performance_from_set_of_recordings(sel_laser_recordings)
    contrasts = (np.array(v.columns).astype('float')*100).astype('int')

    all_v_laser=pd.concat([all_v_laser,v])
    all_a_laser = pd.concat([all_a_laser,a])
    all_c_laser = pd.concat([all_c_laser,c])

    n_v_laser = np.sum([np.sum([k.size for k in rec.vis_choices]) for _,rec in sel_laser_recordings.iterrows()])
    n_a_laser = np.sum([np.sum([k.size for k in rec.aud_choices]) for _,rec in sel_laser_recordings.iterrows()])
    n_c_laser = np.sum([np.sum([k.size for k in rec.coherent_choices]) for _,rec in sel_laser_recordings.iterrows()])
    # NO LASER TRIALS 
    sel_no_laser_recordings= no_laser_df[no_laser_df.Subject==mouse]
    v,a,c = get_performance_from_set_of_recordings(sel_no_laser_recordings)

    all_v_no_laser=pd.concat([all_v_no_laser,v])
    all_a_no_laser = pd.concat([all_a_no_laser,a])
    all_c_no_laser = pd.concat([all_c_no_laser,c])

    n_v_no_laser = np.sum([np.sum([k.size for k in rec.vis_choices]) for _,rec in sel_no_laser_recordings.iterrows()])
    n_a_no_laser = np.sum([np.sum([k.size for k in rec.aud_choices]) for _,rec in sel_no_laser_recordings.iterrows()])
    n_c_no_laser = np.sum([np.sum([k.size for k in rec.coherent_choices]) for _,rec in sel_no_laser_recordings.iterrows()])


fig,ax = plt.subplots(1,3,figsize=(25,7),sharey=True)
ax[0].plot(all_a_laser.mean(),'o-',color='plum',lw=8,markersize=15)
ax[0].plot(all_a_no_laser.mean(),'o-',color='magenta',lw=8,markersize=15)
ax[0].legend(['LED,n=%.0f' % n_a_laser,'no LED, n=%.0f' % n_a_no_laser])
ax[0].set_xlabel('aud azimuth')

ax[1].plot(contrasts,all_v_laser.mean(),'o-',color='lightblue',lw=6,markersize=15)
ax[1].plot(contrasts,all_v_no_laser.mean(),'o-',color='blue',lw=6,markersize=15)
ax[1].legend(['LED,n=%.0f' % n_v_laser,'no LED, n=%.0f' % n_v_no_laser])
ax[1].set_xlabel('contrast')

ax[2].plot(contrasts,all_c_laser.mean(),'o-',color='lightgreen',lw=6,markersize=15)
ax[2].plot(contrasts,all_c_no_laser.mean(),'o-',color='green',lw=6,markersize=15)
ax[2].legend(['LED,n=%.0f' % n_c_laser,'no LED, n=%.0f' % n_c_no_laser])
ax[2].set_xlabel('contrast')
ax[2].set_ylim([-0.05,1.05])
fig.suptitle('%s,%.0f mW, %s hemisphere inhibited, opto+1 (excl. when opto+1=opto)' % (mname,power,inhibited_hemisphere))
# %%
# look at relative RD and relatie nogo on left vs right. RT we only measure at the lowest contrast.
# I think if there is any perceptual change it is most likely to occur on low contrasts

#so I will look at RT and proportion of nogo on the  left vs right
# on ms the low contrast trials 



