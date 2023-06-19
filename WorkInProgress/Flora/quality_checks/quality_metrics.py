# %%

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from pathlib import Path
import numpy as np
import pandas as pd
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json
from Analysis.neural.utils.data_manager import load_cluster_info


from Admin.csv_queryExp import queryCSV


subject_set = ['FT038','FT039','AV024','AV028']
my_expDef = 'all'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

recordings = queryCSV(subject = subject_set,expDate='postImplant',expDef=my_expDef,checkEvents='1',checkSpikes='1',unwrap_independent_probes=True)

curated_rec = recordings[recordings.is_curated]
curated_rec = pd.DataFrame(curated_rec,columns=['subject','expDate','expNum','probe'])

# %%

all_dfs = []
for (_,session) in curated_rec.iterrows():
# get generic info on clusters 
    print(*session)

    ################## MAXTEST #################################
    clusInfo = load_cluster_info(**session)


    all_dfs.append(clusInfo)
    
# temproary hack 
#all_dfs = [d.drop(columns=['sc_azimuth', 'sc_elevation', 'sc_surface']) if 'sc_azimuth' in d.columns else d for d in all_dfs]
clusInfo = pd.concat(all_dfs,axis=0)    


# %%
