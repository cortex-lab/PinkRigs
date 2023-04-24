# %%
# takes histology folder of requested animals and plots their canulla location if it exists 
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

subjects = ['AV029','AV031','AV033'] # list of subjects that we intent to query 
from Admin.csv_queryExp import queryCSV

r = queryCSV(subject=subjects,expDate='last1')
# Add brain regions
# %%
import brainrender as br
scene = br.Scene(title="Canullae locations", inset=False,root=False)