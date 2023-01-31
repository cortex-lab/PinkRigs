# %%
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))


import matplotlib.pyplot as plt
import numpy as np

from Admin.csv_queryExp import load_data
from Analysis.pyutils.wheel_dat import wheel_raster


subject = 'AV031'
data_dict = {
            'events':{'_av_trials':'all'}
                }
recordings = load_data(subject = subject, expDate = '2022-12-12',expDef = 'multiSpace',data_name_dict=data_dict)
# 
ev = recordings.iloc[0].events['_av_trials']


my_wheel = wheel_raster(ev,selected_trials=(ev.is_laserTrial & ev.is_validTrial),align_type='laser')


# %%
