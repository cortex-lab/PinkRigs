# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.loaders import call_neural_dat
from kernel_utils import fit_and_save

from pathlib import Path
import numpy as np

# dataset_name = 'trained-passive-cureated'
# recordings = get_data_bunch(dataset_name) # or can be with queryCSV # or write a loader functin 

ds = 'active' # load the dataset. 
recordings = call_neural_dat(dataset=ds,
                             spikeToInclde=True,
                             camToInclude=True,
                             recompute_data_selection=True,
                             unwrap_probes= True,
                             merge_probes=False,
                             region_selection=None,
                             filter_unique_shank_positions = True,
                             analysis_folder = Path(r'D:\%s' % ds))

# %%
# fit each individual session and save out results. Currently options are: naive, passive, active in dataset type
fit_and_save(recordings,savepath=None,dataset_name=ds,recompute=False,dataset_type='active')

# %%
# reload the results and concatenate to save out a summary 
