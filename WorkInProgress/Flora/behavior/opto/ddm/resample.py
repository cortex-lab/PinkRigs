"""
Model recovery procedure for the ddm model
#todo
-  generation of various samples 2x (train & test sets)
- save 

"""

# %%
import re
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveOpto' 
import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle,resample_model,save_pickle
from fitting import get_parameters
#for subject in subjects:

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\ClusterResults\DriftAdditiveOpto')
sample_path = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\samples\all')
model_path  = basepath 
target_path_train = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\resamples\train')
target_path_test = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\resamples\test')

model_tags = [ 
    'g_both'
    ]

models = sum([list(model_path.glob('*train_Model_%s.pickle' % tag)) for tag in model_tags],[])
n_samples_per_model = 10
for cm in models:
    model = read_pickle(cm)
    # read in the all sample
    cm_sample_path = list(sample_path.glob('%s_Sample*.pickle' % re.split('_Sample_',cm.stem)[0]))[0]
    for i in range(n_samples_per_model):
        train,test = resample_model(model,sample_path=cm_sample_path,split=True)
        # save the sample but idk whether I sohuld just resample here
        save_pickle(train,target_path_train / ('%s_resample_%.0f.pickle' % (cm.stem,i)))
        save_pickle(test,target_path_test / ('%s_resample_%.0f.pickle' % (cm.stem,i)))


  
# %%
