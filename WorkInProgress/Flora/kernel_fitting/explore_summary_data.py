# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Processing.pyhist.helpers.regions import BrainRegions
br = BrainRegions()
from Analysis.pyutils.plotting import brainrender_scattermap

#dat_type = 'AV025AV030AV034postactive'
dat_type = 'naive-allen'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type / 'summary_data.csv'
clusInfo = pd.read_csv(csv_path)
clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position
clusInfo['mlhemi'] = ((clusInfo.ml-5600)*clusInfo.hemi)+5600

clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void'
clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')
clusInfo = clusInfo.dropna(axis=1,how='all') 
# %%
# threshold varaince explained 
thr=0.02
kernel_names = [k for k in clusInfo.keys() if 'kernelVE' in k]
bool_names = []
for k in kernel_names:
    n = k.split('_')[1]
    if len(k.split('_'))==4:
        n = n+'dir'

    if len(k.split('_'))==6:
        n = n+'dir'

    n  = 'is_%s' % (n)
    bool_names.append(n)
    clusInfo[n] = clusInfo[k].values>thr
# %%

from Analysis.neural.utils.spike_dat import call_bombcell_params,bombcell_sort_units
bc_params = call_bombcell_params()


clusInfo = bombcell_sort_units(clusInfo,**bc_params)
clusGood = clusInfo[clusInfo.bombcell_class=='good']

# %%
# plot based just on depth
thr=0.02
fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(clusGood._av_xpos,clusGood.depths,'k.',alpha=0.3)
plt.plot(clusGood._av_xpos[(clusGood['kernelVE_aud_kernel_spl_0.10']>thr)],clusGood.depths[(clusGood['kernelVE_aud_kernel_spl_0.10']>thr)],'m.',alpha=0.3)

# %%

# look at the ratio of neurons per brain region

goodSC = np.array(['SC' in myloc for myloc in clusGood.brainLocationAcronyms_ccf_2017.values])


gSC = clusGood.iloc[goodSC]

# %%
fig,ax = plt.subplots(1,1,figsize=(15,10))

sns.barplot(data = clusGood,x = 'BerylAcronym',y='is_aud',color='magenta',ax=ax)


# %%
fig,ax = plt.subplots(1,1,figsize=(15,10))

orig_df = clusGood


areas,fracs,kernel_type = [],[],[]
for area in np.unique(orig_df.BerylAcronym):
       area_df = orig_df[orig_df.BerylAcronym==area]
       if len(area_df)>20:
              for k in bool_names:
                     areas.append(area)
                     fracs.append(area_df[k].sum()/len(area_df))
                     kernel_type.append(k.split('_')[1])
cellcount_df = pd.DataFrame({
       'brain_area':areas, 
       'proportion_of_cells':fracs,
       'kernel_type': kernel_type
})


if len(bool_names)==3:
     my_hue_order = ['vis','aud']
     my_palette = ['blue','magenta']
elif len(bool_names)==4:
     my_hue_order = ['vis','aud','move','movedir']
     my_palette = ['blue','magenta','black','orange']    

elif len(bool_names)==5:
     my_hue_order = ['vis','aud','auddir','move','movedir']
     my_palette = ['blue','plum','magenta','black','orange']    


sns.barplot(data = cellcount_df,
            x = 'brain_area',
            y='proportion_of_cells',
            hue = 'kernel_type',
            hue_order=my_hue_order,
            palette =my_palette,ax=ax)

ax.set_ylim([0,.6])

off_topspines(ax)
# %%
# plot kernelVE against each other

# df = gSC[['kernelVE_aud',
#        'kernelVE_baseline', 'kernelVE_vis',]]
#g= sns.pairplot(df)


# %%

# import plotly.express as px
# import pandas as pd


# fig = px.scatter(gSC,x='mlhemi', y='dv',color='kernelVE_aud',hover_data=['expFolder','probe','_av_IDs'],range_color=[0,0.05])
# fig.show()


# %%
fig,ax = plt.subplots(1,1,figsize=(4,4))
x = gSC['kernelVE_aud_kernel_spl_0.10_dir']
y = gSC['kernelVE_move_kernel_dir']

ax.plot(x,y,'ko',alpha=0.3)
ax.set_title('r = %.2f' % np.corrcoef(x[(~np.isnan(x)) & (~np.isnan(y))],y[(~np.isnan(x)) & (~np.isnan(y))])[0,1])
ax.set_xlim([-.15,.25])
ax.set_ylim([-.15,.15])
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
off_topspines(ax)


#%%
plt.rcParams.update({'font.size': 22})

fig,ax = plt.subplots(1,1,figsize=(7,7))

#x_s = 'aud_kernel_spl_0.10_dir'
x_s = 'move_kernel_dir'
y_s = 'vis_kernel_contrast_0.40_dir'

x_ve = x_s
y_ve = y_s

if 'vis_kernel_contrast' in x_ve:
      x_ve = 'vis'
if 'vis_kernel_contrast' in y_ve:
      y_ve = 'vis'


gSC['combinedVE'] = gSC[('kernelVE_'+x_ve)] + gSC[('kernelVE_'+y_ve)]
gSC[('kernelSum_'+x_s+'_hemi')] = gSC[('kernelSum_'+x_s)] * gSC.hemi
gSC[('kernelSum_'+y_s+'_hemi')] = gSC[('kernelSum_'+y_s)] * gSC.hemi

sns.scatterplot(data=gSC,
                x=('kernelSum_'+x_s+'_hemi'),
                y=('kernelSum_'+y_s+'_hemi'),
                hue='combinedVE',hue_norm=(-0.05,0.3),legend=False,palette='magma_r',ax=ax)

off_topspines(ax)

# g.axes[0,2].set_xlim((-1,1))
# g.axes[1,2].set_ylim((-1,1))
# %%
import brainrender as br
allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# Add brain regions
which_figure = 'SC_nrns'

if 'all_nrns' in which_figure:
       # previous version
       scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
       scene.add_brain_region("SCs",alpha=0.07,color='sienna')
       sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

       sc = scene.add_brain_region("RN",alpha=0.04,color='teal')
       sc = scene.add_brain_region("MRN",alpha=0.04,color='k')
       sc = scene.add_brain_region("VTA",alpha=0.04,color='y')
       sc = scene.add_brain_region("IC",alpha=0.04,color='y')
       sc = scene.add_brain_region("VISp",alpha=0.04,color='g')

       sc = scene.add_brain_region("PRNr",alpha=0.04,color='r')

       scene.add(br.actors.Points(allen_pos_apdvml, colors='grey', radius=20, alpha=0.3))


       thr = 0.02
       scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_vis>thr)], colors='blue', radius=20, alpha=1))
       scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_aud>thr)], colors='magenta', radius=20, alpha=1))


       scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel>thr)], colors='k', radius=20, alpha=1))
       scene.add(br.actors.Points(allen_pos_apdvml[(clusInfo.kernelVE_move_kernel_dir>thr)], colors='orange', radius=20, alpha=1))

if 'SC_nrns' in which_figure:
       scene = br.Scene(title="%s" % which_figure, inset=False,root=False)
       scene.add_brain_region("SCs",alpha=0.07,color='sienna')
       sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')

       apdvml = gSC[['ap','dv','mlhemi']].values
       apdvml= add_gauss_to_apdvml(apdvml,ml=80,ap=80,dv=0)
       scene.add(br.actors.Points(apdvml, colors='grey', radius=20, alpha=0.3))

       scene.add(br.actors.Points(apdvml[gSC.is_vis], colors='blue', radius=20, alpha=1))
       scene.add(br.actors.Points(apdvml[gSC.is_auddir], colors='magenta', radius=20, alpha=1))
       if len(bool_names)>3:
              scene.add(br.actors.Points(apdvml[gSC.is_move], colors='k', radius=20, alpha=1))
              scene.add(br.actors.Points(apdvml[gSC.is_movedir], colors='orange', radius=20, alpha=1))






scene.render()

# %%
