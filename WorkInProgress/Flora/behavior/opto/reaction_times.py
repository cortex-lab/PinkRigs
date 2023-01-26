# %%

import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_github_paths
add_github_paths()
from Analysis.helpers.queryExp import load_data,Bunch




data_dict = {
    'events':{'_av_trials':'table'}
    }

recdat = load_data(subject = 'AV033',expDate = '2022-12-11', expDef = 'multiSpace',data_name_dict = data_dict)

# %%
