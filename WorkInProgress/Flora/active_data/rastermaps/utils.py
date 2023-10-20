

import numpy as np
from pathlib import Path

def load_dat(session):
    namestring = '{subject}_{expDate}_{expDef}_{probe}'.format(**session)
    savepath = Path(r'C:\Users\Flora\Documents\Processed data\rastermap')
    savepath = savepath / namestring
    savepath.mkdir(parents=True,exist_ok=True)

    R = np.load(savepath / 'spks.npy')
    tscale = np.load(savepath / 'tscale.npy')
    clusIDs  =np.load(savepath / 'clusIDs.npy')
    evs = np.load(savepath / 'evs.npy')
    evnames = np.load(savepath / 'evnames.npy',allow_pickle=True)

    return R,tscale,clusIDs,evs,evnames
