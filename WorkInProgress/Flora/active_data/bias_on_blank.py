# %%
# how many blank trials do I have per session 
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat,Bunch
from Analysis.neural.utils.spike_dat import get_binned_rasters
from Analysis.pyutils.plotting import my_rasterPSTH



mname = 'FT031'
expDate = '2021-12-03'
probe = 'probe0'

plot_kwargs = {
                'pre_time':0.2,
                'post_time':0.3, 
                'bin_size':0.01,
                'smoothing':0.025,
                'return_fr':True,
                'baseline_subtract': False,

        }




ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
other_ = {'events': {'_av_trials': 'table'}}

sess  = 'multiSpaceWorld'



session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
    'probe': probe
}
recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=other_,**session)
if recordings.shape[0] == 1:            
    recordings =  recordings.iloc[0]
else:
    print('recordings are ambiguously defined. Please recall.')


events,spikes,clusters,_,cam = simplify_recdat(recordings,probe='probe')

#%%
clus_ids = recordings.probe.clusters._av_IDs.astype('int')


# for this type of analysis best to look at trials when the perforance was ~50%

nID = 148


keep_trials = (events.is_validTrial & (events.audDiff==0) &
                (events.visDiff==0) &
                ~np.isnan(events.timeline_choiceMoveDir))


t_on  = events.timeline_audPeriodOn[keep_trials]
#t_on = events.timeline_choiceMoveOn[keep_trials]
choices = events.response_direction[keep_trials]
r = my_rasterPSTH(spikes.times,spikes.clusters,[t_on[choices==1],t_on[choices==2]],[nID],event_colors=['blue','red'],**plot_kwargs)




# %%
