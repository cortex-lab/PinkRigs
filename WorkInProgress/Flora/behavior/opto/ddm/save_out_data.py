#%%
#prepare the data into correct pandas format
import sys
import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,concatenate_events

my_subject = ['AV030','AV025','AV034']
recordings = load_data(
    subject = my_subject,
    expDate = '2021-05-02:2023-09-20',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )   

ev = concatenate_events(recordings,filter_type='final_stage')

ev_ = pd.DataFrame.from_dict(ev)

ev_ = ev_.dropna(subset=['rt'])

ev_["response_direction_fixed"] = (ev_["response_direction"]-1).astype(int)

ev_['rt_thresh'] = ev_.timeline_choiceThreshOn-ev_.timeline_audPeriodOn

ev_.to_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv')

# %%
