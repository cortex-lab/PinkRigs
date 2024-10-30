# %%
import glob,sys,itertools
import pandas as pd
import numpy as np
import random
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

from Admin.csv_queryExp import get_csv_location,get_server_list
from brainrender import Scene
from brainrender.actors import Points
# Create a brainrender scene


def add_tracks_to_scene(scene,subject,probe='probe0',mycolor='blue'):
    mainCSV = pd.read_csv(get_csv_location('main'))
    mainCSV = mainCSV[mainCSV.Subject==subject]

    sn = [(mainCSV.P0_serialNo) if '0' in probe else mainCSV.P1_serialNo] 
    sn = float(sn[0].values[0].replace(',', '')) 
    print(sn)
    # read in the mouseList for each mouse 
    servers = get_server_list()
    track_pathstub = r'\%s\histology\registration\brainreg_output\manual_segmentation\standard_space\tracks' % (subject)
    track_list = []
    for s in servers: 
        track_path = s / track_pathstub
        l_server = (list(track_path.glob('%s_SN%.0d_*.npy' % (subject,sn))))
        if len(l_server)>0: 
            [track_list.append(l) for l in l_server]

    # search for the probe tracks and read them in? 

    for track in track_list:
        scene.add(Points(track, colors=mycolor, radius=60, alpha=0.7))

scene = Scene(title="brain regions", inset=True)

# Add brain regions
scene.add_brain_region("SCs",alpha=0.3)
scene.add_brain_region("SCm",alpha=0.3)
#scene.add_brain_region("MRN",color='lightblue',alpha=0.5)
#scene.add_brain_region("CP",color='lightblue',alpha=0.5)
scene.add_brain_region("MOs",alpha=0.3,color = '#024E38')
scene.add_brain_region("VISp",alpha=0.3,color  = '#5EA7DC' )


# scene.add_brain_region("PRNr",alpha=0.8)

subjects = [
   'FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034',
  'AV007','AV013','AV023', 
   'AV008','AV014','AV025','AV030', 
    'AV007','AV013','AV023', 


    ]
probes = [
    'probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0',
   'probe0','probe0','probe0',
    'probe1','probe1','probe1','probe1',
    'probe1','probe1','probe1'

    ]



# Assign a unique color to each subject in a consistent order
subject_colors = {
    'AV005': '#1f77b4',
    'AV008': '#ff7f0e',
    'AV014': '#2ca02c',
    'AV020': '#d62728',
    'AV025': '#9467bd',
    'AV030': '#8c564b',
    'AV034': '#e377c2',
    'FT030': '#7f7f7f',
    'FT031': '#8c564b',
    'FT032': '#bcbd22',
    'FT035': '#17becf',
    'AV007': '#9e9e9e',
    'AV009': '#f7b6d2',
    'AV013': '#c5b0d5',
    'AV015': '#c49c94',
    'AV021': '#e5b6a8',
    'AV023': '#f8e0a1'
}


# Generate the final list of colors matching the length of subjects list
colors = [subject_colors[subject] for subject in subjects]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
unique_subjects = list(set(subjects))

# Plot each subject with its corresponding color
for i, subject in enumerate(unique_subjects):
    color = subject_colors[subject]
    plt.gca().add_patch(mpatches.Rectangle((0, i), 1, 1, facecolor=color, edgecolor='black'))
    plt.text(1.5, i + 0.5, subject, verticalalignment='center', fontsize=12)

plt.xlim(0, 3)
plt.ylim(0, len(unique_subjects))
plt.axis('off')  # Turn off the axis
plt.show()


for s,p,c in zip(subjects,probes,colors):
    print(s,p,c)
    add_tracks_to_scene(scene,s,probe=p,mycolor=c)

scene.render()






# %


# anim = Animation(scene, "./examples", "vid3")

# # Specify camera position and zoom at some key frames
# # each key frame defines the scene's state after n seconds have passed
# anim.add_keyframe(0, camera="top", zoom=1)
# anim.add_keyframe(1.5, camera="sagittal", zoom=0.95)
# anim.add_keyframe(3, camera="frontal", zoom=1)
# anim.add_keyframe(4, camera="frontal", zoom=1.2)

# # Make videos
# anim.make_video(duration=5, fps=15)
# %%
