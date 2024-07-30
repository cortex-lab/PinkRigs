
# %%
import sys
import numpy as np
import pandas as pd 
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches

# pinkRig modules
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import queryCSV
from Processing.pyhist.helpers.atlas import AllenAtlas
from Processing.pyhist.helpers.regions import BrainRegions
from Analysis.neural.utils.spike_dat import anatomy_plotter
from opto_utils import get_relative_eYFP_intensity



atlas,br = AllenAtlas(25),BrainRegions()

subjects = ['AV029','AV031','AV033','AV036','AV038','AV041','AV044','AV046','AV047'] # list of subjects that we intent to query 

#subjects = ['AV036']

recordings = queryCSV(subject=subjects,expDate='last1')

stub = r'Histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]

# save summary anatomical data: subject,ap,dv,ml,hemisphere(-1:Left,1:Right),regionAcronym 

data = pd.DataFrame()
for idx,m in enumerate(histology_folders):
    cannulae_list = list(m.glob('*.npy'))
    for c in cannulae_list:
        subject = m.parents[5].name
        track = np.load(c)
        # canulla tip point (because I always start tracking at the tip)
        tip_ccf = track[0]
        # assert the position of these tip points in allen atlas space location
        region_id = atlas.get_labels(atlas.ccf2xyz(track[0],ccf_order='apdvml'))
        region_acronym=br.id2acronym(region_id) # get the parent of that 

        # calculate the relative fluorescence for each track...
        rel_fluorescence = get_relative_eYFP_intensity(c)
        
        if subject=='AV029':
            rel_fluorescence = np.nan
            print(subject,'did not have eYFP')
            

        data = data.append(
            {'subject':subject,
            'ap':tip_ccf[0], 
            'dv':tip_ccf[1],
            'ml':tip_ccf[2], 
            'hemisphere':-int(np.sign(tip_ccf[2]-5600)), 
            'region_id':region_id, 
            'region_acronym':region_acronym[0],
            'parent1':br.acronym2acronym(region_acronym, mapping='Beryl')[0],
            'eYFP_fluorescence': rel_fluorescence
            },ignore_index=True
        )


# plot 
anat = anatomy_plotter()
_, (ax,ax1) = plt.subplots(1,2,figsize=(15,4))

# the top view
anat.plot_anat_canvas(ax=ax,axis = 'dv',coord = 1400)
anat.plot_points(data.ml, data.ap,s=50,color='red',alpha=1,unilateral=True)
anat.plot_points(data.ml, data.ap,s=1000,color='red',alpha=.1,unilateral=True)
rectangleL = matplotlib.patches.Rectangle([-1450,-4075],800,4075-3800,alpha=0.2)
rectangleR = matplotlib.patches.Rectangle([650,-4075],800,4075-3800,alpha=0.2)

ax.add_patch(rectangleL)
ax.add_patch(rectangleR)

ax.set_ylim([-5000,-2600])
ax.set_xlim([-2200,2200])

# the coronal section
anat.plot_anat_canvas(ax=ax1,axis = 'ap',coord = 3700)
anat.plot_points(data.ml, data.dv,s=50,color='red',alpha=1,unilateral=True)
anat.plot_points(data.ml, data.dv,s=1000,color='red',alpha=.1,unilateral=True)
ax1.set_xlim([-2200,2200])
ax1.set_ylim([-3000,-800])

plt.suptitle('%s' % subject)


plt.show()

which_figure = 'cannulae_map_%s' % ('all')
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


#%%
# save an distance from RF thing
data.to_csv(r'D:\opto_cannula_locations.csv')
# %%
