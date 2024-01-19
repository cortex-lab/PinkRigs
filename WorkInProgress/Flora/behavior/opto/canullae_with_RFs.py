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

dat_type = 'naive-acute-rfs'
dat_keys = get_data_bunch(dat_type)

#from Admin.csv_queryExp import queryCSV
#recordings = queryCSV(subject=['AV028'],expDate='2022-10-26',expDef='sparseNoise',checkSpikes='1',checkEvents='1')
# dat_keys = recordings[['subject','expDate','expNum']]
# dat_keys['probe']='probe0'
# %%
csv_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\%s\%s' % (dat_type,'summary_data.csv'))
recompute = False 

all_dfs = []
if csv_path.is_file() & (not recompute):
    clusInfo = pd.read_csv(csv_path)
else:
    for _,session in dat_keys.iterrows():
    # get generic info on clusters 
        print(*session)
        clusInfo = load_cluster_info(**session)
        rf = rf_model(**session)
        rf.fit_evaluate()

        # some clusters have no spikes so we drop those from clusInfo
        _,idx,_ = np.intersect1d(clusInfo._av_IDs,rf.score.neuronID,return_indices=True)
        clusInfo = clusInfo.iloc[idx]

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

from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.pyutils.plotting import brainrender_scattermap

allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)



score_thr = 0.05

# %%
fig,ax = plt.subplots(1,1,figsize=(2,10))
my_hemi = -1
ax.hist(clusInfo.depths[(clusInfo.hemi==my_hemi)],orientation='horizontal',bins=50,color='grey',alpha=0.5)
ax.hist(clusInfo.depths[(clusInfo.score>score_thr) & (clusInfo.hemi==my_hemi)],orientation='horizontal',bins=25,alpha=0.7,color='lightseagreen')
off_topspines(ax)
ax.set_xlabel('# neurons')
ax.set_ylabel('distance from tip (um)')
ax.legend(['all','visRFs'])



# %%
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

subjects = ['AV029','AV031','AV033','AV036','AV038','AV041','AV044','AV046','AV047'] # list of subjects that we intent to query 

#subjects = ['AV041']
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


# swap everything to one hemisphere
fig,ax = plt.subplots(1,1,figsize=(10,5))
ap=allen_pos_apdvml[:,0]
ml = allen_pos_apdvml[:,2]

azimuth = clusInfo.fit_azimuth
swap_single_hemi = True
if swap_single_hemi: 
    allen_pos_apdvml_r = clusInfo[['ap','dv','ml']].values
    allen_pos_apdvml_r[:,2] =  ((clusInfo.ml-5600)*clusInfo.hemi+5600)
    azimuth = azimuth *clusInfo.hemi
    data.ml = (data.ml-5600) * data.hemisphere * -1 +5600


#allen_pos_apdvml_r= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)

which=(clusInfo.score>score_thr) & (azimuth<90) & (azimuth>0)
plt.scatter(ml[which],ap[which],c=azimuth[which],vmin=0,vmax=90,cmap='Blues')
plt.colorbar()

plt.scatter(data.ml,data.ap,s=70,c='red')
plt.xlabel('ml,SC')
plt.ylabel('ap')

# 
fig,ax = plt.subplots(1,1,figsize=(10,5))
elevation = clusInfo.fit_elevation
which=(clusInfo.score>score_thr)# & (azimuth<15) & (azimuth>-15)
plt.scatter(ml[which],ap[which],c=elevation[which],vmin=-25,vmax=25,cmap='Greys')
plt.colorbar()

plt.scatter(data.ml,data.ap,s=70,c='red')
plt.xlabel('ml,SC')
plt.ylabel('ap')

# %%
from Processing.pyhist.helpers.atlas import AllenAtlas
from Analysis.pyutils.plotting import brainrender_scattermap
from scipy import stats 

atlas = AllenAtlas(25)

# bar plots for SCs and SCm
# for the spatial cells
_,ax = plt.subplots(1,1,figsize=(5,5))
p = allen_pos_apdvml
xyz = atlas.ccf2xyz(p,ccf_order='apdvml') 
t= 'aud'
clusInfo['is_SC'] = ['SC'in loc for loc in clusInfo.brainLocationAcronyms_ccf_2017.astype('str')]

mm = np.nanmin(xyz[clusInfo.is_SC & (clusInfo.score>score_thr),2])
ma  = np.nanmax(xyz[clusInfo.is_SC & (clusInfo.score>score_thr),2])

requested_depth = 2000
xyz_ref = atlas.ccf2xyz(np.array([allen_pos_apdvml[0,0],requested_depth,allen_pos_apdvml[0,2]]),ccf_order='apdvml')

atlas.plot_hslice(xyz_ref[2],volume='boundary',ax=ax,aspect='auto')

# ax.set_ylim([-5000,-2600])
# ax.set_xlim([-2200,0])

#ax.scatter(-dots_to_plot[:,2]+5600,-dots_to_plot[:,0]+5400,color=dot_colors,edgecolor='grey',s=100,alpha=.8,vmin=0,vmax=.3) 

ax.scatter(-data.ml+5600,-data.ap+5400,s=50,color='red')
ax.scatter(-data.ml+5600,-data.ap+5400,s=1000,color='red',alpha=.1)


is_plotted= (clusInfo.score>score_thr) & clusInfo.is_SC
xfromprobe = clusInfo._av_xpos-clusInfo._av_shankID*200

dots_to_plot = allen_pos_apdvml_r[is_plotted,:]

dot_colors = brainrender_scattermap(elevation[is_plotted],vmin = -37.5,vmax=37.5,n_bins=5,cmap='coolwarm')

#ax.scatter(-dots_to_plot[:,2]+5600+xfromprobe[is_plotted],-dots_to_plot[:,0]+5400+xfromprobe[is_plotted],color=dot_colors,edgecolor='grey',s=100,alpha=.8,vmin=0,vmax=.3) 
ax.set_ylim([-5000,-2600])
ax.set_xlim([-2200,0])

which_figure = 'cannulae_map'
cpath  = Path(r'C:\Users\Flora\Pictures\PaperDraft2024')
im_name = dat_type + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg
plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# %%
# linear model for ap vs azimuth 


# %%
