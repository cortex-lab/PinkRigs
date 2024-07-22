
#%%
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data


ephys_dict = {'spikes': 'times'}
recordings = load_data(data_name_dict={'probe0': ephys_dict, 'probe1': ephys_dict,'events':{'_av_trials':'table'}},
                     subject='CB019', expDate='2021-10-25',expDef='all',
                     expNum=None,
                     checkSpikes='1',
                     unwrap_probes=False, merge_probes=False)





#%%
ephys_dict = {'spikes':'all','clusters':'all'}
ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 


cam_dict = {'frontCam':{'camera':['times','ROIMotionEnergy']},
                           'eyeCam':{'camera':['times','ROIMotionEnergy']},
                           'sideCam':{'camera':['times','ROIMotionEnergy']}}



recordings = load_data(data_name_dict = None,
                        subject = ['FT008','FT009','FT010','FT011','FT019','FT022','FT025','FT027','FT038','FT039'],
                        expDef = 'AV', 
                        #expDef = ['AVPassive_spatialIntegrationFlora_with_ckecker',,'AVPassive_ckeckerboard_extended','AVPassive_checkerboard_extended'],
                        unwrap_probes=True, merge_probes=False,filter_unique_shank_positions=True,cam_hierarchy=None)



['AVPassive_spatialIntegrationFlora_with_ckeckerboard',
       'AVPassive_ckeckerboard_updatechecker',
       'AVPassive_ckeckerboard_extended',
       'AVPassive_checkerboard_extended']


# %%
