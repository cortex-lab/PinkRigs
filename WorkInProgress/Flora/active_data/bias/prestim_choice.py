# this script calculates the difference in firing rate in training vs test set for eac trial type within a recording 
# it has the potentail to take into account different time bins 
# and add different trial types/subsample the trial types 
#%%
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data
from Analysis.neural.utils.data_manager import load_cluster_info
from dPrime import get_choicePrime

subject_set = ['AV008','AV014','AV020','AV025','AV030','AV034']
my_expDef = 'multiSpaceWorld'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

# load sessions and recordings that are relevant

ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = subject_set,expDate='postImplant',
                        expDef=my_expDef,
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.6})
#{'region_name':'SC','min_fraction':.3}
# %%
triggers = ['timeline_audPeriodOn','timeline_choiceMoveOn']
t = .2

recordings['choicePrime'] = [get_choicePrime(rec,t=t,rt_min=0.05,contrasts='low4', onset_names= triggers,plot_summary=True,plot_nrns='top5') for _, rec in recordings.iterrows()]

     
# %%
# make data long form with each cluster
#clusInfo = pd.concat([load_cluster_info(rec,probe='probe') for _,rec in recordings.iterrows()]) 
# concatenate clusInfo & train + test set for this thing

# %% 
all_clus = []
for _,rec in recordings.iterrows():
    clusInfo = load_cluster_info(rec,probe='probe')
    for trigger in triggers: 
        stim = rec.choicePrime[trigger]
        if stim is not None:
            clusInfo['%s_train' % trigger] = stim.train 
            clusInfo['%s_test' % trigger] = stim.test
        else: 
            clusInfo['%s_train' % trigger] = np.nan
            clusInfo['%s_train' % trigger] = np.nan
    all_clus.append(clusInfo)

clusInfo = pd.concat(all_clus)
# %%

# plot the  train vs test for the various brain regions or something? 
# introduce quality controls and region controls

from Analysis.neural.utils.spike_dat import bombcell_sort_units
from Processing.pyhist.helpers.regions import BrainRegions
br = BrainRegions()

bc_class = bombcell_sort_units(clusInfo)
clusInfo['is_good'] = bc_class=='good'

clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void' # this is just so that the berylacronymconversion does something good

clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')

# %%
import matplotlib.pyplot as plt
plt.hist(clusInfo.BerylAcronym[clusInfo.is_good])
# get the various brain regions 
# check if each region contains a min no. of good neurons
# plot those regions that do

rois = np.unique(clusInfo.BerylAcronym[clusInfo.is_good])
n_units_per_roi = np.array([np.sum((clusInfo.BerylAcronym==i) & (clusInfo.is_good) & (~np.isnan(clusInfo.timeline_audPeriodOn_train))) for i in rois])
min_n = 100
my_rois = rois[(n_units_per_roi>min_n) & (rois!='root')  & (rois!='void')]

n_regions = len(my_rois)
s_unit = 2.5
fig,axs = plt.subplots(2,n_regions,figsize=(n_regions*s_unit,2*s_unit),sharex=True,sharey=True)


for ridx,c_roi in enumerate(my_rois):
    is_selected = (clusInfo.BerylAcronym==c_roi) & clusInfo.is_good & ~np.isnan(clusInfo.timeline_audPeriodOn_train)
    for tidx,trigger in enumerate(triggers): 
        choicePrime_train = clusInfo['%s_train' % trigger][is_selected]
        choicePrime_test = clusInfo['%s_test' % trigger][is_selected]
        hemisphere = clusInfo.hemi[is_selected]
        ax = axs[tidx,ridx]
        if 'PeriodOn' in trigger:
            tstring = 'stim'
        else: 
            tstring = 'movement'
        cmin,cmax = np.nanmin([choicePrime_train,choicePrime_test]),np.nanmax([choicePrime_train,choicePrime_test])
        #cmin,cmax = cmin*1.1,cmax*1.1
        c = np.max(np.abs([cmin,cmax]))*1.1
        c=70
        ax.scatter(choicePrime_train,choicePrime_test,c=hemisphere,cmap='coolwarm',edgecolor='k',vmin=-1,vmax=1,alpha=.5)
        ax.set_xlim([-c,c])
        ax.set_ylim([-c,c])
        ax.axline((0,0),slope=1,color='k',linestyle='--')
        ax.text(-c*.8,c*0.8,'r=%.2f' % (np.corrcoef(choicePrime_train,choicePrime_test)[0,1]))            
        
        if tidx==0:
            ax.set_title('%s,%.0f neurons' % (c_roi,is_selected.sum()))
        
        if ridx==0: 
            ax.set_ylabel('pre-%s'% tstring)


fig.suptitle('R-L on blank trials,train vs test set, good neurons only')
# %%
