# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.loaders import call_neural_dat
from kernel_utils import fit_and_save



# dataset_name = 'trained-passive-cureated'
# recordings = get_data_bunch(dataset_name) # or can be with queryCSV # or write a loader functin 

recordings = call_neural_dat(dataset='active',
                             spikeToInclde=True,
                             camToInclude=True,
                             recompute_data_selection=True,
                             unwrap_probes= True,
                             merge_probes=False,
                             region_selection=None,
                             filter_unique_shank_positions = True)

# %%
fit_and_save(recordings,savepath=None,dataset_name='active',recompute=True,dataset_type='active')

# %%
