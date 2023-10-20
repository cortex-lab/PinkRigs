from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

namestring = '{subject}_{expDate}_{expDef}_{probe}'.format(**session)
savepath = Path(r'C:\Users\Flora\Documents\Processed data\rastermap')
savepath = savepath / namestring

measures =  np.load(savepath / 'spt.npy')
behav =  np.load(savepath / 'behav.npy')

from rastermap import Rastermap, utils
from scipy.stats import zscore
measures = zscore(measures, axis=1)

nancells = np.isnan(measures).all(axis=1)

measures = measures[~nancells,:]

model = Rastermap(n_PCs=50, n_clusters=None, 
                  locality=.15, time_lag_window=0,grid_upsample=0).fit(measures)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
#X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)


fig,ax = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [6, 1]})
ax[0].matshow(measures[y[:,0].astype('int'),:],aspect='auto',vmin=-5,vmax=5,cmap='coolwarm')
ax[1].plot(behav)
plt.show()

