#%%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')

#%%
subject_set = ['AV025','AV030','AV034']
expDef_sets = ['postactive','multiSpaceWorld_checker_training']

d = {}
for my_expDef in expDef_sets:
    subject_string = ''.join(subject_set)
    dataset = subject_string + my_expDef


    csv_path = interim_data_folder / dataset 
    fit_tag = 'stimChoice'
    foldertag = r'kernel_model\%s' % fit_tag
    kernel_fit_results = interim_data_folder / dataset  / foldertag 


    subfolders = [item for item in kernel_fit_results.iterdir() if item.is_dir()]

    recordings = [tuple(s.name.split('_')) for s in subfolders]
    recordings = pd.DataFrame(recordings,columns=['subject','expDate','expNum','probe'])

    d[my_expDef] = recordings

# %%

# select the animal-day-probe sets that are present for both 
passive = (d[expDef_sets[0]]).drop(labels='expNum',axis=1)
active = (d[expDef_sets[1]]).drop(labels='expNum',axis=1)


matched = passive.merge(active, how = 'inner' ,indicator=False,left_index=True)
# %%



