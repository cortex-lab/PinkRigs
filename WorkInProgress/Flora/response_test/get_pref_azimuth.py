# this code will: 
#determine whether a nrn has azimuthal tuning 
# and determine the preferred azimuth 
# and for neurons that do not have a preferred azimuth: will fake a prefferred azimuth based on location
# and calculate the MS index for them.

# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
session = { 
    'subject':'FT009',
    'expDate': '2021-01-20',
    'expNum': 7
}
azi = azimuthal_tuning(session)







# %%
