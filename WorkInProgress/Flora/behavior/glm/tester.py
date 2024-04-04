
#%%
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data



recordings = load_data(data_name_dict = {'frontCam':{'camera':['times','ROIMotionEnergy']},
                           'eyeCam':{'camera':['times','ROIMotionEnergy']},
                           'sideCam':{'camera':['times','ROIMotionEnergy']}},
                        subject = ['AV030'],
                        unwrap_probes=False, merge_probes=False,cam_hierarchy=['sideCam','eyeCam','frontCam'])

# %%
