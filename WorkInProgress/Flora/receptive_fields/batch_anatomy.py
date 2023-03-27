# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines

from Analysis.neural.utils.data_manager import load_cluster_info,write_cleanCSV
from Analysis.neural.src.rf_model import rf_model

dat_type = 'naive-receptiveField'
from Admin.csv_queryExp import queryCSV
# 
recordings = queryCSV(subject='AV024',expDate='2022-10-11',expDef='sparseNoise',checkSpikes='1')
dat_keys = recordings[['subject','expDate','expNum']]
dat_keys['probe']='probe0'
csv_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\%s\%s' % (dat_type,'summary_data.csv'))
recompute = True 
# %%
all_dfs = []
if csv_path.is_file() & (not recompute):
    clusInfo = pd.read_csv(csv_path)
else:
    for _,session in dat_keys.iterrows():
    # get generic info on clusters 
        clusInfo = load_cluster_info(**session)
        rf = rf_model(**session)
        rf.fit_evaluate()

        clusInfo['score'] = rf.score.sel(cv_number=1).values

        azi,elevation,a_sig,e_sig = rf.get_rf_degs_from_fit()
        clusInfo['fit_azimuth'] = azi
        clusInfo['fit_sigma_azimuth'] = a_sig
        clusInfo['fit_elevation'] = elevation
        clusInfo['fit_sigma_elevation'] = e_sig

        

        all_dfs.append(clusInfo)
    clusInfo = pd.concat(all_dfs,axis=0)
    write_cleanCSV(clusInfo,csv_path)

# %%
# prepare positional values
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.pyutils.plotting import brainrender_scattermap

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)

score_thr = 0.05
dots_to_plot = allen_pos_apdvml[clusInfo.score>score_thr]
dot_colors = brainrender_scattermap(clusInfo.fit_sigma_azimuth.values[clusInfo.score>score_thr],vmin = 0,vmax=15,n_bins=7,cmap='coolwarm')


# %%
# Add brain regions
from brainrender import Scene
from brainrender.actors import Points
scene = Scene(title="SC aud and vis units", inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.05,color='grey')
sc = scene.add_brain_region("SCm",alpha=0.05,color='grey')

scene.add(Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.8))

# for p,c in zip(dots_to_plot,dot_colors):
#    scene.add(Points(p[np.newaxis,:]), colors=c, radius=30, alpha=0.8)

scene.content
scene.render()
# %%
