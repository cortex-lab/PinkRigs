# %%
# takes histology folder of requested animals and plots their canulla location if it exists 
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

subjects = ['AV029','AV031','AV033','AV036','AV038'] # list of subjects that we intent to query 
from Admin.csv_queryExp import queryCSV

recordings = queryCSV(subject=subjects,expDate='last1')

stub = r'Histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
histology_folders = [
    (Path(r.expFolder).parents[1] / stub) for _,r in recordings.iterrows()
]
# Add brain regions
# %%
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

scene.render()


# %%
