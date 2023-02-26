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
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
session = { 
    'subject':'FT009',
    'expDate': '2021-01-20',
    'expNum': 7,
    'probe': 'probe0'
}
azi = azimuthal_tuning(session)

# %% 
# developing permutation test to get significant tuning

# to get significant tuning we will correlate cv split and shuffle the labels.
tuning_type = 'vis'
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': None, # means I select the max
    'which': tuning_type,
    'subselect_neurons':None,
}

# %%
# test for azimuthal selectivity

azi.get_rasters_perAzi(**tuning_curve_params)
is_selective,preferred_tuning = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)


# %%
