

"""
prototyping a function that spits out recordings that have neural data in a requested brain areas
"""
#%%
import sys,shutil
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import numpy as np
import pandas as pd
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json


from Admin.csv_queryExp import queryCSV,load_ephys_independent_probes,load_data


subject_set = ['AV008','AV020','AV014','AV025','AV030','AV034']
my_expDef = 'postactive'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

recordings = load_data(data_name_dict = {'probe0':{'clusters':'all'},'probe1':{'clusters':'all'}},
                        subject = subject_set,expDate='postImplant',
                        expDef=my_expDef,
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})



# %%
