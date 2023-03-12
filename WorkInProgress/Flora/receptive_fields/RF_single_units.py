# %%

# this code fits receptive fields of individual units

import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes
from Analysis.neural.src.rf_fit import rf_fit

ephys_dict =  {'spikes': ['times', 'clusters']}
other_ = {'events':{'_av_trials':'all'}}

recordings = load_ephys_independent_probes(
    subject = 'FT009',
    expDate='2021-01-20',
    expDef = 'AP_sparseNoise',
    checkSpikes = '1',  
    probe = 'probe0',
    ephys_dict=ephys_dict,
    add_dict=other_
    )

rec = recordings.iloc[0]
sn_info = rec.events._av_trials

rf = rf_fit()
rf.add_sparseNoise_info(sn_info)



# %%
