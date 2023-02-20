# this code will: 
#determine whether a nrn has azimuthal tuning 
# and determine the preferred azimuth 
# and for neurons that do not have a preferred azimuth: will fake a prefferred azimuth based on location
# and calculate the MS index for them.

# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning,get_test_statistic
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
    'cv_split': 2,
    'subselect_neurons':None,
    'azimuth_shuffle': None
}

tc = azi.get_tuning_curves(**tuning_curve_params)
actual_ = get_test_statistic(tc)

# tuning clolumns
def sample_null_dist(tc_params,my_seed): 
    tc_params['azimuth_shuffle'] = my_seed
    tc = azi.get_tuning_curves(**tc_params)
    null_statistic = get_test_statistic(tc)
    return null_statistic

# %%
