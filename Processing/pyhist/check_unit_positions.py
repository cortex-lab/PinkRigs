## scripts to check unit positions in the atlas 

# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from Admin.csv_queryExp import queryCSV,load_data
from Analysis.neural.utils.data_manager import get_sessions_with_units,get_highest_yield_unique_ephys

#recordings = queryCSV(subject = 'AV034',expDate='2022-12-07')
my_probe = 'probe0'

data_dict = {
'probe0':{'clusters':'all'},
'probe1':{'clusters':'all'},
'probe0_raw':{'channels':'localCoordinates'},
'probe1_raw':{'channels':'localCoordinates'}
}

#kwargs['expDef'] = expdef_namestring
recdat = load_data(subject='AV005', expDate = '2022-05-11:2022-05-13',data_name_dict = data_dict)
recdat = recdat[(recdat.extractSpikes=='1')]
#recordings = get_sessions_with_units(subject='AV034', expDef = 'postactive')

# recordings = recordings[(recordings.extractEvents=='1') & (recordings.extractSpikes=='1')]
# recordings  = recordings[['subject','expDate','expNum']]
# %%
sess_idx = 1
rec = recdat.iloc[sess_idx]
probe = rec[my_probe].clusters
ch = rec[(my_probe+'_raw')].channels

# %
import matplotlib.pyplot as plt
fig,ax  = plt.subplots(1,1)
ax.hist(ch.localCoordinates[:,1],alpha=.3,color='k')
ax.hist(probe.depths,color = 'lightseagreen')


# %%
fig,ax  = plt.subplots(1,1)

for _,rec in recdat.iterrows():
    probe = rec[my_probe].clusters
    dv = probe.mlapdv[:,2]
    ax.plot(probe.depths,dv,'.')

# %%


import brainrender as br
from Analysis.pyutils.plotting import brainrender_scattermap



# %%
# Add brain regions
which_figure = 'all_nrns'
scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.07,color='sienna')
sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

sc = scene.add_brain_region("RN",alpha=0.04,color='teal')
sc = scene.add_brain_region("MRN",alpha=0.04,color='k')
sc = scene.add_brain_region("VTA",alpha=0.04,color='y')


n_recordings = len(recdat)
colors = brainrender_scattermap(np.arange(n_recordings),n_bins=n_recordings,cmap = 'Set1') 
for (_,rec),c in zip(recdat.iterrows(),colors):
    probe = rec[my_probe].clusters

    scene.add(br.actors.Points(probe.mlapdv[:,[1,2,0]], colors=c, radius=20, alpha=0.3))


scene.render()




# %%
