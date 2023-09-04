# %%
#  for each mice that have 10 & 17mW 
# calculate deltaRT for each power
# for both ipsi and contralatearal choices
#

import sys
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_topspines
from Analysis.pyutils.ev_dat import getTrialNames

my_subject = ['AV036','AV038','AV041','AV044','AV046','AV047']
recordings = load_data(
    subject = my_subject,
    expDate = '2022-06-02:2023-08-01',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'all'}}
    )

#
ev,_,_,_,_ = zip(*[simplify_recdat(rec,reverse_opto=True) for _,rec in recordings.iterrows()])
is_laser_session = [np.sum(e.is_laserTrial)>0 for e in ev]
ev = list(compress(ev,is_laser_session))
# 
ev_keys = list(ev[0].keys())
ev = Bunch({k:np.concatenate([e[k] for e in ev]) for k in ev_keys})



# %
# %%
