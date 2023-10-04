#%%
#prepare the data into correct pandas format
import sys
import pandas as pd

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,concatenate_events

my_subject = ['AV036']
recordings = load_data(
    subject = my_subject,
    expDate = '2021-05-02:2023-09-04',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )   

dat_type = 'optoUniBoth'
ev = concatenate_events(recordings,filter_type=dat_type)

ev_ = pd.DataFrame.from_dict(ev)


ev_["response_direction_fixed"] = (ev_["response_direction"]-1).astype(int)

ev_['rt_thresh'] = ev_.timeline_choiceThreshOn-ev_.timeline_audPeriodOn
ev_['rt_laserThresh'] = ev_.timeline_choiceThreshPostLaserOn-ev_.block_laserStartTimes

if 'opto' in dat_type: 
    ev_ = ev_.dropna(subset=['rt_laserThresh'])
else:
    ev_ = ev_.dropna(subset=['rt','rt_thresh'])

# get rid of invalid trials,nogos (should not be any at this point 
ev_ = ev_[ev_.is_validTrial & ((ev_.laser_power==17)|(ev_.laser_power==0))]

ev_.to_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\Optomice_10mWunilateral.csv')
