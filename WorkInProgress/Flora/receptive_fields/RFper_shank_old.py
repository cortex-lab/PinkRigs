# %%
import sys,glob,os 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt 
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
import Analysis.neural.src.rf_model as rf 
from Admin.csv_queryExp import load_data
from Analysis.pyutils.plotting import off_axes 
subject = 'FT030'
probe = 'probe0'
data_dict = {'events':{'_av_trials':['squareAzimuth','squareElevation','squareOnTimes']},probe:{'spikes':['times','clusters','depths','_av_shankIDs']}}
recordings = load_data(subject = subject, expDate = '2021-12-01',data_name_dict=data_dict,expDef='sparseNoise')
rec = recordings.iloc[0]
spikes = rec[probe].spikes
sn_info = rec.events._av_trials

# %% 


findRF = rf.rf_model()
findRF.format_events(sn_info)
findRF.bin_spikes_per_loc(spikes)
a =findRF.binned_spikes_depths['array']
keep_shank = (a.sum(axis=-1).sum(axis=-1)) > 0 

s = a[keep_shank,:,:][0,:,:]


fig,ax = plt.subplots(s.shape[0],1, figsize=(1,15))
for d in range(s.shape[0]): 

    r =findRF.get_response_binned(s[d,:])
    vmax=np.max(np.abs(r))/3
    ax[-d].imshow(r,aspect='auto',vmin=-vmax,vmax=vmax,cmap='coolwarm')
    off_axes(ax[-d])
# %%
