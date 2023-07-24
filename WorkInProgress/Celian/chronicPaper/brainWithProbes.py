# -*- coding: utf-8 -*-

from brainrender import Scene
from brainrender.actors import Points
from paper.figures import INSET# removes the scale bars?
from myterial import black

dataPath = "//znas/Lab/Share/Celian/ChronicPaper/brainrender/"
screenFolder = dataPath + "screenshots"

# create scene
scene = Scene(atlas_name="allen_mouse_25um",inset=INSET,screenshots_folder=screenFolder)
scene.root._needs_silhouette = False

# add brain regions
scene.add_brain_region('MOs','NDB', alpha=0.4)
# scene.add_brain_region('CP','MOp','SSp','SSs','GPe','GPi', alpha=0.4, hemisphere='right')

# add probe tracks
folder_tracks = "X:/AV009/histology/registration/brainreg_output/manual_segmentation/standard_space/tracks/" 
import glob
trackList = glob.glob(folder_tracks + "AV*.npy")
#trackList = glob.glob(folder_tracks + "*19011119461*.npy")

for p in range(len(trackList)):
    file = trackList[p]
    scene.add(Points(file, colors=black, radius=30, alpha=0.7))
    
cam = {
       "pos": (0,-36430, -5700),
       "viewup": (-1, 0, 0),
       "clippingRange": (32024, 63229),
       "focalPoint": (7319, 2861, -3942),
       "distance": 43901,
    }

scene.render(interactive=True, camera=cam, zoom=1.5)