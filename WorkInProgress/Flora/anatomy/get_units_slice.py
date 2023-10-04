# %%
# scripts to call an experiment and plot the location of the units in atlas space
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data
from Processing.pyhist.helpers.atlas import AllenAtlas
atlas = AllenAtlas(25)

mname = 'AV014'
expDate = '2022-06-16'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
}

d = {'probe0':{'clusters':'all'}, 
     'probe1':{'clusters':'all'}, 
} 

r = load_data(data_name_dict=d,**session)


fig,ax = plt.subplots(1,1,figsize=(15,10))
c = r.iloc[0].probe0.clusters
c1 = r.iloc[0].probe1.clusters

if len(c)>0:
    p = c.mlapdv
    xyz = atlas.ccf2xyz(p,ccf_order='mlapdv') 
    atlas.plot_cslice(np.nanmean(xyz[:,1]),volume='boundary')
    ax.plot(p[:,0]-5600,-p[:,2],'r.')
    if len(c1)>0:
        p1 = c1.mlapdv
        ax.plot(p1[:,0]-5600,-p1[:,2],'r.')
else: 
    print('no spikes..')


 # %%
