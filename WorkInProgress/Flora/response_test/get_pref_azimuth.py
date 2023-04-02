
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
tuning_curves = azi.fit_evaluate(cv_split=2,metric='svd')
# %%
testedID = 56
azi.plot_response_per_azimuth(neuronID=testedID,which='p')
azi.plot_tuning_curves(tuning_curves=tuning_curves,neuronID=testedID)

# %%
# alternative methods to test: 
# A) subtract average PC1 from actual response
import numpy as np
import matplotlib.pyplot as plt

neuron_idx = np.where(azi.clus_ids==testedID)[0][0]

r = azi.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
[u,s,v] = np.linalg.svd(r_,full_matrices=False)
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
neuron_idx = np.where(azi.clus_ids==testedID)[0][0]

r = azi.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
[u,s,v] = np.linalg.svd(r_,full_matrices=False)
m,n = u.shape[0], u.shape[1]
s_ = np.zeros((m,n))
s_[0:len(s),0:len(s)] = np.diag(s)

fromPC=0
toPC = 2
r_pc = u[:,fromPC:toPC] @ np.diag(s[fromPC:toPC]) @ v[fromPC:toPC,:] 
r_pc = r_pc.reshape(r.shape)
fig,ax = plt.subplots(1,7,sharey=True,figsize=(15,4))
for i,a in enumerate(r_pc.mean(axis=1)):
    ax[i].plot(a)



# %% method 3   take the trial weight from the 1st pc 
neuron_idx = np.where(azi.clus_ids==testedID)[0][0]

r = azi.response_rasters_per_azimuth.dat[:,:,neuron_idx,:]
r_ = np.reshape(r,(r.shape[0]*r.shape[1],r.shape[2]))
[u,s,v] = np.linalg.svd(r_,full_matrices=False)
tc = np.reshape(u[:,0],(7,30))
fig,ax = plt.subplots(1,1)
ax.plot(np.abs(tc.mean(axis=1)))

# %%
is_selective,preferred_tuning = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)

# %%

# non-linearity test as performed by Standford et al.,2005
# at the preferred azimuth only.

# actually one has the data already loaded with azi 
# so now, get response at preferred azi 

azi.get_enhancement_index()

