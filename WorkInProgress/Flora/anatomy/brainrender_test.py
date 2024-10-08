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
    sn = float(mainCSV.P0_serialNo.values[0].replace(',','')) 

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
scene.add_brain_region("SCs",alpha=0.6)
scene.add_brain_region("SCm",alpha=0.6)
scene.add_brain_region("MRN",color='lightblue',alpha=0.8)
# scene.add_brain_region("PRNr",alpha=0.8)

subject_list = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034']
subject_list = ['AV007','AV009','AV013','AV015','AV021','AV023']

subjects = [
    'FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034',
    'AV007','AV009','AV013','AV015','AV021','AV023', 
    'AV008','AV014','AV020','AV025','AV030', 
    'AV007','AV009','AV013','AV015','AV023', 


    ]
probes = [
    'probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0','probe0',
    'probe0','probe0','probe0','probe0','probe0','probe0',
    'probe1','probe1','probe1','probe1','probe1',
    'probe1','probe1','probe1','probe1','probe1',

    ]

unique_subjects = list(set(subjects))

# Generate unique hex colors for each unique subject
subject_colors = {subject: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for subject in unique_subjects}

# Assign the same color to 'probe0' and 'probe1'
probe_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
subject_colors['probe0'] = probe_color
subject_colors['probe1'] = probe_color

# Generate the final list of colors matching the length of subjects list
colors = [subject_colors[subject] for subject in subjects]


for s,p,c in zip(subjects,probes,colors):
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
