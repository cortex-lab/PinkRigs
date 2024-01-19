#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.spike_dat import get_binned_rasters

subject_set = ['AV025']
my_expDef = 'multiSpaceWorld'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

# load sessions and recordings that are relevant

ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = subject_set,expDate='2022-11-09',
                        expDef=my_expDef,
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})


# %%
rec = recordings.iloc[0]
ev,spikes,clusters,_,_ = simplify_recdat(rec,probe='probe')


#%%
raster_kwargs = {
        'pre_time':0.4,
        'post_time':0, 
        'bin_size':0.050,
        'smoothing':0,
        'return_fr':True,
        'baseline_subtract': False, 
}




keep_l =   ( ev.is_validTrial & 
    ~np.isnan(ev.timeline_choiceMoveOn) &
    (ev.response_direction==1)) & (ev.rt>0.1)
keep_r =   ( ev.is_validTrial & 
    ~np.isnan(ev.timeline_choiceMoveOn) &
    (ev.response_direction==2)) & (ev.rt>0.1)


t_on_l = ev.timeline_choiceMoveOn[keep_l]

t_on_r = ev.timeline_choiceMoveOn[keep_r]



r = get_binned_rasters(spikes.times,spikes.clusters,[0],t_on_r[np.argsort(ev.rt[keep_r])],**raster_kwargs)

plt.imshow(np.abs(np.diff(r.rasters[:,0,:])),aspect='auto')
# %%
from Analysis.pyutils.plotting import my_rasterPSTH

plot_kwargs = {
'pre_time':1,
'post_time':.1, 
'bin_size':0.01,
'smoothing':0.025,
'return_fr':True,
'baseline_subtract': False,'rasterlw':2.5,'n_rasters':500}

nID=21
fig,(ax,ax1) = plt.subplots(2,1,figsize=(6,12))
r = my_rasterPSTH(spikes.times,spikes.clusters,[t_on_l[np.argsort(ev.rt[keep_l])],t_on_r[np.argsort(ev.rt[keep_r])]],[nID],event_colors=['blue','red'],ax=ax,ax1=ax1,**plot_kwargs)

# %%
