# %% 
import sys,shutil
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import numpy as np
import pandas as pd
from kernel_utils import fit_and_save


from Admin.csv_queryExp import load_data


subject_set = ['AV043']
data_name_dict = { 'events': {'_av_trials': 'table'}}
cam_dict = {'frontCam':{'camera':['times','ROIMotionEnergy']},
                        'eyeCam':{'camera':['times','ROIMotionEnergy']},
                        'sideCam':{'camera':['times','ROIMotionEnergy']}}
ephys_dict = {'spikes':'all','clusters':'all'}
ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 

data_name_dict.update(ephys_dict)
data_name_dict.update(cam_dict)

recordings = load_data(subject = subject_set,data_name_dict=data_name_dict,
                       expDate='postImplant',
                       expDef='multiSpaceWorld',
                       checkEvents='1',checkSpikes='1',unwrap_probes=True)

fit_and_save(recordings,savepath=None,dataset_name='dominance',recompute=True,dataset_type='active')


# %%

# %%
