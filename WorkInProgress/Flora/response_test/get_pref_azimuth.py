# this code will: 
#determine whether a nrn has azimuthal tuning 
# and determine the preferred azimuth 
# and for neurons that do not have a preferred azimuth: will fake a prefferred azimuth based on location
# and calculate the MS index for them.

# %% 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning,get_discriminability,get_tc_correlations
session = { 
    'subject':'FT009',
    'expDate': '2021-01-20',
    'expNum': 7
}
azi = azimuthal_tuning(session)

# %% 
# developing permutation test to get significant tuning

# to get significant tuning we will correlate cv split and shuffle the labels.
tuning_type = 'vis'

tuning_curve_params = { 
    'contrast': 1,
    'spl': None, 
    'which': tuning_type,
    'subselect_neurons':None,
}

# %%
# test for discriminability

azi.get_rasters_perAzi(**tuning_curve_params)
tc = azi.get_tuning_curves(cv_split=1,azimuth_shuffle_seed=None)
d_actual = get_discriminability(tc)

def sample_null_dist(my_seed):
    tc = azi.get_tuning_curves(cv_split=1,azimuth_shuffle_seed=my_seed)
    null_statistic = get_discriminability(tc)
    return null_statistic

n_shuffles = 1000
d_shuff = [sample_null_dist(shuffle_idx)[np.newaxis,:] for shuffle_idx in range(n_shuffles)]
d_shuff = np.concatenate(d_shuff,axis=0)
# calculate p value
d_actual_ =np.tile(d_actual,(n_shuffles,1))
p_val = (d_shuff>d_actual_).sum(axis=0)/n_shuffles

plt.hist(p_val)
# also get preferred tuning after 5 cv splits 
# %%
tc_pref = azi.get_tuning_curves(cv_split=5,azimuth_shuffle_seed=None)
c_actual = get_tc_correlations(tc_pref)
# tuning clolumns
def sample_null_dist(my_seed):
    tc = azi.get_tuning_curves(cv_split=5,azimuth_shuffle_seed=my_seed)
    null_statistic = get_tc_correlations(tc)
    return null_statistic

n_shuffles = 1000
c_shuff = [sample_null_dist(shuffle_idx)[np.newaxis,:] for shuffle_idx in range(n_shuffles)]
c_shuff = np.concatenate(c_shuff,axis=0)
# calculate p value
c_actual_ =np.tile(c_actual,(n_shuffles,1))
p_val = (c_shuff>c_actual_).sum(axis=0)/n_shuffles
# 

                                                                                                                                                                                                                                                                     # %%

# %%
# see how it does for some neurons 


clusID = 50
cidx = np.where(azi.clus_ids==clusID)[0][0]
_,ax = plt.subplots(1,1)
ax.hist(d_shuff[:,cidx])
ax.axvline(c_actual[cidx])


# %%
