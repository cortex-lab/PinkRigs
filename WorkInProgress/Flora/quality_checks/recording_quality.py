# %%
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
import numpy as np 
import matplotlib.pyplot as plt 

subject = 'AV025'
expDate = '2022-11-07:2022-11-08'
probe = 'probe1'
raw_probe = probe + '_raw'
data_dict = {probe:{'clusters':'all'},raw_probe:{'clusters':'all'}}
recordings = load_data(subject = subject,expDate= expDate,expDef='sparseNoise',data_name_dict=data_dict)
#%%
are_spikes = ['depths' in rec[probe].clusters.keys() for _,rec in recordings.iterrows()]
recordings = recordings[are_spikes]
# %%


def plot_cluster_locations(rec,probe,ax,only_good=True): 
    clusters = rec[probe].clusters
    if only_good: 
        goodclus_index = np.where(rec[probe].clusters.ks2_label=='good')[0]
    else: 
        goodclus_index = np.array(list(range(0,clusters.depths.size)))
    ax.plot(clusters._av_xpos[goodclus_index],
            clusters.depths[goodclus_index],'o',
            color='k',alpha=.5,markersize = 5)

fig,ax = plt.subplots(1,1,figsize=(10,5))
[plot_cluster_locations(rec,probe,ax,only_good=False) for _,rec in recordings.iterrows()]
ax.set_xlabel('xpos')
ax.set_ylabel('depth')
plt.show()
# %%



