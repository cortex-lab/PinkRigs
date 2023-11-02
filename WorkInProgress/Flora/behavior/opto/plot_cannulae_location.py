# %%
# takes histology folder of requested animals and plots their canulla location if it exists 
import sys
import numpy as np
import pandas as pd 
from pathlib import Path

# pinkRig modules
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import queryCSV
from Processing.pyhist.helpers.atlas import AllenAtlas
from Processing.pyhist.helpers.regions import BrainRegions

atlas,br = AllenAtlas(25),BrainRegions()

#subjects = ['AV029','AV031','AV033','AV036','AV038','AV041'] # list of subjects that we intent to query 

subjects = ['AV041']
recordings = queryCSV(subject=subjects,expDate='last1')

stub = r'Histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]

# %%

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

        data = data.append(
            {'subject':subject,
            'ap':tip_ccf[0], 
            'dv':tip_ccf[1],
            'ml':tip_ccf[2], 
            'hemisphere':-int(np.sign(tip_ccf[2]-5600)), 
            'region_id':region_id, 
            'region_acronym':region_acronym[0],
            'parent1':br.acronym2acronym(region_acronym, mapping='Beryl')[0]},ignore_index=True
        )

# save this as a file
#data.to_csv(r'C:\Users\Flora\Documents\Processed data\Audiovisual\cannula_locations.csv')




# %%
# show tracks in brainrender
import brainrender as br
from Analysis.pyutils.plotting import brainrender_scattermap
n_mice = len(histology_folders)
mouse_colors = brainrender_scattermap(np.arange(n_mice),vmin=-1,vmax=n_mice-1,cmap='Accent',n_bins=n_mice)

scene = br.Scene(title="Cannulae locations", inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.07,color='sienna')
sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

for idx,m in enumerate(histology_folders):
    cannulae_list = list(m.glob('*.npy'))
    for c in cannulae_list:
        track = np.load(c)
        scene.add(br.actors.Points(track, colors=mouse_colors[idx], radius=60, alpha=0.5))
        scene.add(br.actors.Points(track[0][np.newaxis,:], colors=mouse_colors[idx], radius=120, alpha=0.5))

interValue = True 
pltView = 'coronal'
pltSlice = True
if pltSlice:
    scene.slice("frontal")



if pltView == "coronal":
    cam = {
        "pos": (-36430, 0, -5700),
        "viewup": (0, -1, 0),
        "clippingRange": (40360, 64977),
        "focalPoint": (7319, 2861, -3942),
        "distance": 43901,
    }
elif pltView == "side":
    cam = {
        "pos": (11654, -32464, 81761),
        "viewup": (0, -1, -1),
        "clippingRange": (32024, 63229),
        "focalPoint": (7319, 2861, -3942),
        "distance": 43901,
    }

scene.render(interactive=interValue,camera=cam,zoom=3.5)


# %%
