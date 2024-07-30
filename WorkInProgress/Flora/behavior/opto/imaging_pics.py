# %%

import sys,os
import numpy as np
import pandas as pd 
from pathlib import Path

import matplotlib.pyplot as plt

# image related modules 
from skimage import io
from skimage.util import img_as_float


# pinkRig modules
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import queryCSV
from Analysis.pyutils.plotting import off_axes



# 
def normcocktail(r,g,epsilon=.1):
    cocktail=np.sqrt(r**2+g**2)
    r_c=r/(cocktail+epsilon)
    g_c=g/(cocktail+epsilon)
    return r_c,g_c


# image processing functions 
def norm_image(im,e=0.0000001): 
    im = img_as_float(im)
    norm_im=(im-np.min(im))/(np.max(im)-np.min(im)+e)
    return norm_im


subjects = ['AV029','AV031','AV033','AV036','AV038','AV041','AV044','AV046','AV047'] 

subject = ['AV031']

recordings = queryCSV(subject=subject,expDate='last1')
stub = r'Histology\registration\brainreg_output'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]



basepath  = histology_folders[0]

rsource=os.path.join(basepath,'downsampled_standard.tiff')
gsource=os.path.join(basepath,'downsampled_standard_green.tiff')

r=io.imread(rsource)
g=io.imread(gsource)
#%%
track_path = basepath / 'manual_segmentation/standard_space/tracks'
cannulae_list = list(track_path.glob('*.npy'))
tips = [np.load(c)[0]/25 for c in cannulae_list]

        # canulla tip point (because I always start tracking at the tip)
# %%
_,ax = plt.subplots(1,1,figsize=(15,10))


idx = int(np.floor(tips[0][0]))
e_norm, e_cocktail = 1000,1000
mycmap = 'pink'
x1,x2 = 25,200 
y1,y2 = 100,350
#r[idx,x1:x2,y1:y2]
im_r,im_g = norm_image(r[idx,:,:],e=e_norm), norm_image(g[idx,:,:],e=e_norm)
im_r_,im_g_ = normcocktail(im_r,im_g,epsilon=e_cocktail)
ax.matshow(im_r_,cmap=mycmap,alpha=.9,aspect='auto')
ax.matshow(im_g_,cmap=mycmap,alpha=.9,aspect='auto')
[ax.plot(tip[2],tip[1],'o') for tip in tips]

#ax.set_title(subject)
#off_axes(ax)
#ax.invert_xaxis()

# get the cannula tip location in atlas space


which_figure = '%s_SChist_%.0f' % (subject,idx)
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#ax.set_title('%s_right' % subject)

# %%
