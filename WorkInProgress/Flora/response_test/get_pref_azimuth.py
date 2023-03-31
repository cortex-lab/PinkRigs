
# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
session = { 
    'subject':'FT009',
    'expDate': '2021-01-20',
    'expNum': 8,
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
tuning_curves = azi.fit_evaluate(cv_split=2)
# %%
azi.plot_response_per_azimuth(neuronID=22,which='p')

# %%
# alternative methods to test: 
# A) subtract average PC1 from actual response
import numpy as np
import matplotlib.pyplot as plt

neuron_idx = np.where(azi.clus_ids==66)[0][0]

r = azi.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
[u,s,v] = np.linalg.svd(r_)
nPC=2
r_pc = u[:,:nPC] @ np.diag(s[:nPC]) @ v[:nPC,:] # 1st PC
r_pc = r_pc.mean(axis=0)
fig,ax = plt.subplots(1,7,sharey=True)
tc = []
for i,a in enumerate(r.mean(axis=1)): 
    ax[i].plot(a-r_pc)
    tc.append((a-r_pc).max())

fig,ax = plt.subplots(1,1)
ax.plot(tc)
# %%
# yet another alternative: we reconstruct from PC2<
neuron_idx = np.where(azi.clus_ids==66)[0][0]

r = azi.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
[u,s,v] = np.linalg.svd(r_)
fromPC=1
r_pc = u[:,fromPC:] @ np.diag(s[fromPC:]) @ v[fromPC:,:] 

# %%
is_selective,preferred_tuning = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)


# %%

azi.plot_tuning_curves(tuning_curves=tuning_curves,neuronID=66)


# %%

# non-linearity test as performed by Standford et al.,2005
# at the preferred azimuth only.

# actually one has the data already loaded with azi 
# so now, get response at preferred azi 

azi.get_enhancement_index()

