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

dat_type = 'naive-allen'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position


which_figure = 'kernelVE-me'

# %% 
# look at discriminability 
 
# plotting in brainrender 
import brainrender as br
import numpy as np

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# Add brain regions
if 'responsive-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good & clusInfo.is_SC,:], colors='grey', radius=20, alpha=0.7))
    scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good & clusInfo.is_SC,:], colors='b', radius=20, alpha=0.7))
    scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good & clusInfo.is_SC,:], colors='m', radius=20, alpha=0.7))
    scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_both & clusInfo.is_good & clusInfo.is_SC,:], colors='g', radius=20, alpha=0.7))
elif 'vis-spatial-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    t = 'vis'
    dots_to_plot = allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t]),:]
    dot_colors = brainrender_scattermap(clusInfo['x0%s' % t][clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])],vmin = -90,vmax=90,n_bins=15,cmap='coolwarm')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))

elif 'aud-spatial-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    t = 'aud'
    dots_to_plot = allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t]),:]
    dot_colors = brainrender_scattermap(clusInfo['x0%s' % t][clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])],vmin = -90,vmax=90,n_bins=15,cmap='coolwarm')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))

elif 'movement-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_movement_correlated.astype('bool') & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.movement_correlation)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.movement_correlation[is_plotted],vmin = 0,vmax=0.5,n_bins=15,cmap='Greys')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))

elif 'kernelVE-vis' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_vis)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_vis[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='Blues')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))    

elif 'kernelVE-aud' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_aud)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_aud[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='PuRd')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))  

elif 'kernelVE-me' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_motionEnergy)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_motionEnergy[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='Greys')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))  


scene.render()
