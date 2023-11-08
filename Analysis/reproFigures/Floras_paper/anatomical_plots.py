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

dat_type = 'naive-total'
#dat_type = 'trained-passive-cureated'

interim_data_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position


which_figure = 'kernelVE-vis'
one_hemisphere = True


# look at discriminability 
 
# plotting in brainrender 
import numpy as np

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
radii = 50
alpha = .3
if one_hemisphere:
    allen_pos_apdvml[:,2] = np.abs(allen_pos_apdvml[:,2]-5600)+5600



from Processing.pyhist.helpers.atlas import AllenAtlas
atlas = AllenAtlas(25)
thr=.02
p = allen_pos_apdvml
xyz = atlas.ccf2xyz(p,ccf_order='apdvml') 
fig,ax = plt.subplots(1,1,figsize=(3,3))

p[:,1] = p[:,1]-200
is_plotted = clusInfo.is_good & clusInfo.is_SC 
ax.scatter(p[is_plotted,2]-5600,-p[is_plotted,1],color='grey',edgecolor='none',s=16,alpha=.2,vmin=0,vmax=.3)
vmax=.05
from matplotlib.colors import ListedColormap


grey = [.1,.1,.1,1]
green =[.5,1,0,1]
my_cmap = np.linspace(grey,green,356)

cmap = ListedColormap(my_cmap)

dotsize = 50

if 'me' in which_figure:
    is_plotted = clusInfo.is_good & clusInfo.is_SC & (clusInfo.kernelVE_motionEnergy>thr)
    ax.scatter(p[is_plotted,2]-5600,-p[is_plotted,1],c=clusInfo.kernelVE_motionEnergy[is_plotted],edgecolor='k',cmap=cmap,s=dotsize,alpha=.8,vmin=0,vmax=vmax)
if 'aud' in which_figure:
    is_plotted = clusInfo.is_good & clusInfo.is_SC & (clusInfo.kernelVE_aud>thr)
    ax.scatter(p[is_plotted,2]-5600,-p[is_plotted,1],c=clusInfo.kernelVE_aud[is_plotted],edgecolor='k',cmap=cmap,s=dotsize,alpha=.8,vmin=0,vmax=vmax)
if 'vis' in which_figure: 
    is_plotted = clusInfo.is_good & clusInfo.is_SC & (clusInfo.kernelVE_vis>thr)
    plt.scatter(p[is_plotted,2]-5600,-p[is_plotted,1],c=clusInfo.kernelVE_vis[is_plotted],edgecolor='k',cmap=cmap,s=dotsize,alpha=.8,vmin=0,vmax=vmax)

atlas.plot_cslice(np.mean(xyz[is_plotted,1]),volume='boundary',ax=ax,aspect='auto')

ax.set_xlim([0,2400])
ax.set_ylim([-3200,-600])

cpath  = Path(r'C:\Users\Flora\Pictures\SfN2023')
im_name = dat_type + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

#%%
import brainrender as br

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
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=alpha))

elif 'aud-spatial-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    t = 'aud'
    dots_to_plot = allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t]),:]
    dot_colors = brainrender_scattermap(clusInfo['x0%s' % t][clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])],vmin = -90,vmax=90,n_bins=15,cmap='coolwarm')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=alpha))

elif 'movement-neurons' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_movement_correlated.astype('bool') & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.movement_correlation)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.movement_correlation[is_plotted],vmin = 0,vmax=0.5,n_bins=15,cmap='Greys')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=0.5))

elif 'kernelVE-vis' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_vis)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_vis[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='Blues')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=alpha))    

elif 'kernelVE-aud' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_aud)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_aud[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='PuRd')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=alpha))  

    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_aud) & clusInfo.is_aud
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_aud[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='PuRd')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=1))  

elif 'kernelVE-me' in which_figure:
    scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
    scene.add_brain_region("SCs",alpha=0.07,color='sienna')
    sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
    is_plotted = clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo.kernelVE_motionEnergy)
    dots_to_plot = allen_pos_apdvml[is_plotted,:]
    dot_colors = brainrender_scattermap(clusInfo.kernelVE_motionEnergy[is_plotted],vmin = 0,vmax=0.1,n_bins=15,cmap='Greys')
    scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=radii, alpha=alpha))  


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
