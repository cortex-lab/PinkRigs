# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveOpto' 
import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle



refit_options = [
        'ctrl',
        'drift_bias',
        'sensory_drift',
        'starting_point',
        'mixture',
        'nondectime',
        'all'
    ]

#for subject in subjects:

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'
sample_path = basepath / 'Data/forMyriad/samples/train'

datasets = list(sample_path.glob('*.pickle'))

#%%

model_path = (model_path / ('%s_Model_%s.pickle' % (sample_path.stem,type)))
