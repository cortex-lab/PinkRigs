# setting paths 
#%%
import sys,itertools
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.io import add_github_paths
add_github_paths()

#
import pandas as pd 
import numpy as np 
from src.maxtest import maxtest as m
maxtest = m()

from Analysis.helpers.queryExp import load_data
from utils.data_manager import simplify_recdat
from utils.ev_dat import postactive
probe = 'probe0'
rec_info = pd.Series(
    ['FT009','2021-01-20',7],
    index = ['subject','expDate','expNum']
    )


# load data
recordings = load_data(data_name_dict=maxtest._av_required_data,**rec_info)
events,spikes,_,_ = simplify_recdat(recordings.iloc[0])
blank,vis,aud,ms = postactive(events)


event_dict = {}
# to redo the previous tests
checked_spl = np.max(aud.SPL.values)
# for (azimuth,d_power) in itertools.product(aud.azimuths.values,[checked_spl]):
#     stub = 'aud_azimuth_%.0d_dpower_%.2f' % (azimuth,d_power)
#     event_dict[stub] = aud.sel(azimuths=azimuth,SPL=d_power,timeID='ontimes').values
for (azimuth,d_power) in itertools.product(vis.azimuths.values,vis.contrast.values):
    stub = 'vis_azimuth_%.0d_dpower_%.2f' % (azimuth,d_power)
    event_dict[stub] = vis.sel(azimuths=azimuth,contrast=d_power,timeID='ontimes').values   

# for (azimuth,spl,contrast) in itertools.product(ms.congruent_azimuths[0],[checked_spl],ms.contrast.values):
#     stub = 'ms_azimuth_%.0d_spl_%.2f_%.2f' % (azimuth,spl,contrast)
#     event_dict[stub] = ms.sel(visazimuths=azimuth,audazimuths=azimuth,SPL=spl,contrast=contrast,timeID='audontimes').values


p = maxtest.run(
    spikes,
    event_dict,
    blank.sel(timeID='ontimes').values,
    subselect_neurons=None,
    n_shuffles=2000,
    plotting = True
    )
bonferroni_p_thr = 0.01/p.columns.size
is_signifiant = (p<bonferroni_p_thr).any(axis=1).to_numpy()
# %%
