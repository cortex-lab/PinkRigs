# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.pyutils.plotting import brainrender_scattermap

dat_type = 'AV025postactiveprobe1'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position

# %%
import brainrender as br
allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# Add brain regions
which_figure = 'all_nrns'
scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.07,color='sienna')
sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

sc = scene.add_brain_region("RN",alpha=0.04,color='teal')
sc = scene.add_brain_region("MRN",alpha=0.04,color='k')
sc = scene.add_brain_region("VTA",alpha=0.04,color='y')

scene.add(br.actors.Points(allen_pos_apdvml, colors='grey', radius=20, alpha=0.3))


thr = 0.02
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_vis>thr)], colors='blue', radius=20, alpha=1))
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_aud>thr)], colors='magenta', radius=20, alpha=1))


scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel>thr)], colors='k', radius=20, alpha=1))
scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel_dir>thr)], colors='orange', radius=20, alpha=1))

scene.render()
