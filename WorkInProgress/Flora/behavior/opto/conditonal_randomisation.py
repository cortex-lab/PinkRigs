# 
import sys
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_topspines
from Analysis.pyutils.ev_dat import getTrialNames

my_subject = ['AV047']
recordings = load_data(
    subject = my_subject,
    expDate = '2023-06-02:2023-07-27',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )
