
# %% 

from utils import load_dat

mname = 'AV030'
expDate = '2022-12-11'
probe = 'probe0'
sess='multiSpaceWorld'

session = { 
    'subject':mname,
    'expDate': expDate,
    'expDef': sess,
    'probe': probe
}


R,tscale,clusIDs,evs,evnames = load_dat(session)
# %%
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap, utils
from scipy.stats import zscore



blstim = evs[evnames=='baselinePeriod'] + evs[evnames=='stimPeriod']

keepIdx  = np.where(blstim[0]>0)[0]

# spks is neurons by time
spks = zscore(R, axis=1)
#%%
spks_ = spks[:,keepIdx]
evs_ = evs[:,keepIdx]
# fit rastermap
model = Rastermap(n_PCs=50, n_clusters=50, 
                  locality=0.0, time_lag_window=0).fit(spks_)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
#X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)


fig,ax = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [6, 1]})
ax[0].matshow(spks_[isort,:],aspect='auto',vmin=0,vmax=1,cmap='Greys')
ax[1].plot(evs_.T)
plt.show()
# basically we got to digitise the events as well
# %%
