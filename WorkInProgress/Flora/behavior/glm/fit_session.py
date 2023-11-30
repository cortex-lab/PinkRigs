
#%%
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data

recordings = load_data(data_name_dict = {'events': {'_av_trials': 'table'}},
                        subject = 'AV030',expDate='2022-12-08',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=False,
                        region_selection=None)


# %%
from predChoice import format_av_trials,glmFit

ev = recordings.iloc[0].events._av_trials

trials = format_av_trials(ev)
glm = glmFit(trials,cv_type=None)
glm.caluclatepHat(model_type='AVSplit')
