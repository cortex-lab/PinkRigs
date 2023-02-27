
# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
session = { 
    'subject':'FT009',
    'expDate': '2021-01-20',
    'expNum': 7,
    'probe': 'probe0'
}
azi = azimuthal_tuning(session)

tuning_type = 'aud'
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': 0.02, # means I select the max
    'which': tuning_type,
    'subselect_neurons':None,
}

azi.get_rasters_perAzi(**tuning_curve_params)
is_selective,preferred_tuning = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)



# %%
