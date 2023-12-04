
#%%
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = 'AV030',expDate='2022-12-14',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=False,
                        region_selection=None)




# %%
from predChoice import format_av_trials,glmFit

ev = recordings.iloc[0].events._av_trials
spk = recordings.iloc[0].probe1.spikes

trials = format_av_trials(ev,spikes=spk)


# %%
glm = glmFit(trials.iloc[:,:3],model_type='AVSplit')
#glm.fit()
glm.fitCV(n_splits=5,test_size=0.3)
glm.visualise(yscale='sigmoid')
# %%
