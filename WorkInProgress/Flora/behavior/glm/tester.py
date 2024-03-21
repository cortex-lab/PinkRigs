
#%%
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data



ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict},
                        subject = ['AV043'],expDate='2024-03-14',
                        expNum='3',
                        checkSpikes='1',
                        unwrap_probes=False, merge_probes=False)

# %%
