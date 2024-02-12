# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.batch_data import get_data_bunch
from kernel_utils import fit_and_save



dataset_name = 'trained-passive-cureated'
recordings = get_data_bunch(dataset_name) # or can be with queryCSV
fit_and_save(recordings,savepath=None,dataset_name=dataset_name,recompute=True,dataset_type='passive')

# %%
