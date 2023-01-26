# %%
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
import pandas as pd
import numpy as np
from pathlib import Path

from utils.data_manager import get_data_bunch, load_cluster_info

from src.azimuthal_tuning import azimuthal_tuning
from src.AVmodel_psth import default_fitting


save_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
dataset = 'naive'
kernels_path = save_path / dataset / 'kernel_model'

recs = get_data_bunch(dataset)

clusters = []
tc = []
for idx,rec_info in recs.iterrows():
    # get the tuning curve
    tuning_curve = azimuthal_tuning(rec_info=rec_info)
    ta = tuning_curve.get_tuning_curves(which = 'aud',cv_split=2)
    tv = tuning_curve.get_tuning_curves(which = 'vis',cv_split=2)

    ta_cv2 = ta[ta.cv_number==1]
    tv_cv2 = tv[tv.cv_number==1]
    ta_cv2 = ta_cv2.drop('cv_number',axis=1)
    tv_cv2 = tv_cv2.drop('cv_number',axis=1)

    ta = ta[ta.cv_number==0]
    tv = tv[tv.cv_number==0]
    ta = ta.drop('cv_number',axis=1)
    tv = tv.drop('cv_number',axis=1)



    # get the model fits 
    ve = default_fitting(rec_info,repeats=3)
    # get the cluster information. 
    clusInfo = load_cluster_info(**rec_info)
    # load the kernel fitting results
    # kernel_results = pd.read_csv((kernels_path / ('%s_%s_%.0f_%s.csv' % tuple(rec_info))))
    # # requires serious pivoting in the current format
    # cluster_info_test = kernel_results[(kernel_results.cv_number==1)]
    # df = cluster_info_test[['VE','clusID','event']]
    # df = df.reset_index(drop=True)
    # df = df.pivot(index=['clusID'],columns='event')
    # df = df.VE
    # df = df[['aud','motionEnergy','vis']]

    rec_clusters_info = pd.concat([clusInfo,ta,tv,ve],axis = 1)
    rec_clusters_info['recording_ID']  = '%s_%s_%s_%s' % tuple(rec_info)  # just a unique identification for the recordings's information. 
    
    tc.append(pd.concat([ta_cv2,tv_cv2],axis=1))
    clusters.append(rec_clusters_info)


clusters = pd.concat(clusters).reset_index()
#  calculate registration
clusters['depths_from_sc'] = clusters.depths-clusters.sc_surface 


   # %%
# plot location of each unit compared to the SC
import plotly.express as px
goodclus = clusters[
    (clusters.slidingRP_viol == 1) & 
    (clusters.noise_cutoff>20) & 
    (clusters.depths_from_sc.notna()) & 
    (clusters.sc_azimuth.notna()) & 
    (clusters.winner_model.notna()) & 
    (clusters.spike_count>100) &
    (clusters.depths_from_sc>-1400) &
    (clusters.depths_from_sc<0)
    ]

goodclus = clusters[
    (clusters._av_KSLabels ==2) & 
    (clusters.depths_from_sc>-1400) &
    (clusters.depths_from_sc<0)
]

fig = px.scatter(
    goodclus,
    x='sc_azimuth', 
    y='depths_from_sc',
    color='winner_model',
    hover_data=['recording_ID','_av_IDs']
    )
fig.show()


# %% 

def histogram_selected(x,ix,myc,ax,ori='horizontal',mybins=50):
    ax.hist(x[ix],range=(-1400,0),bins=mybins,orientation=ori,color=myc,histtype='stepfilled',lw=4)
    # if x[ix].size/mybins>10:
    #     ax.set_xlim([0,150])   
    # if x[ix].size/mybins>7:
    #     ax.set_xlim([0,150]) 
    # else: 
    #     ax.set_xlim([0,20])

import matplotlib.pyplot as plt
from utils.plotting import off_topspines,off_exceptx,off_axes,off_excepty
myMorder=['baseline','V_spatial','A_spatial','V_spatial_A_spatial',['full','nonlinear'],'A_center','V_spatial_A_center']
colors=['grey','blue','magenta','green','black','plum','lightgreen']

_,ax = plt.subplots(2,len(myMorder),figsize=(8,8),gridspec_kw= dict(height_ratios=[2,3]))
#ax[0,1].set_yticks([0,-400,-850,-1400])

for mix,model in enumerate(myMorder):
    if type(model)==list:
        ix=np.concatenate([np.where((goodclus['winner_model']==m))[0] for m in model])
    else:
        ix=np.where((goodclus['winner_model']==model))[0]
    myc=colors[mix]
    histogram_selected(goodclus['depths_from_sc'].values,ix,myc,ax[1,mix],mybins=20)
    ax[0,mix].bar(0,(ix.size/goodclus.shape[0])*100,color=myc)
    if mix>0:
        off_exceptx(ax[1,mix])
        off_axes(ax[0,mix])
        #ax[1,mix].set_xlim([0,10])
    else:
        off_topspines(ax[1,mix])
        ax[1,mix].set_xlim([0,20])
        off_excepty(ax[0,mix])
    ax[0,mix].set_ylim([0,50])
    #ax[1,0].set_xlim([0,5])
    ax[1,3].set_xlim([0,5])
    ax[1,4].set_xlim([0,5])
    ax[1,5].set_xlim([0,5])
    ax[1,6].set_xlim([0,5])

plt.savefig("C:\\Users\\Flora\\Pictures\\totwinnermodels_trained.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
model_winner_clus = goodclus[
    (clusters.winner_model == 'V_spatial_A_spatial')]

#model_winner_clus['ap_hemispheric'] = model_winner_clus.ap*np.sign(goodclus.ml-5600)


# %%
model_winner_clus['preferred_vis_tuning'] = model_winner_clus.preferred_vis_tuning.astype('float64')
model_winner_clus['preferred_aud_tuning'] = model_winner_clus.preferred_aud_tuning.astype('float64')

sns.histplot(
    model_winner_clus,
    x = 'sc_azimuth', 
    y = 'preferred_aud_tuning',
    bins=(7,7),
    color = 'magenta'
    )

# %%
#sns.pairplot(goodclus[['aud','vis','motionEnergy']])
# curation really helps btw 


# look at the spatial maps of each of these neurons

# kernelclus = goodclus[(goodclus.vis<0.002) & (goodclus.aud>0.002)]

sns.histplot(
    kernelclus,
    x = 'sc_azimuth', 
    y = 'preferred_aud_tuning',
    bins=(7,7),
    color = 'magenta'
    )
# %%
clusters['preferred_aud_tuning'] = clusters.preferred_aud_tuning.astype('float64')
clusters['preferred_vis_tuning'] = clusters.preferred_vis_tuning.astype('float64')

def minmax_norm_df(df):
    """
    function to normalise the values of a dataframe in each row
    designed to normalise tuning curves 

    """
    df_cols = df.columns
    df = df.to_numpy()
    minval = np.tile(df.min(axis=1)[:,np.newaxis],df.shape[1])
    minmax_val = np.tile((df.max(axis=1)-df.min(axis=1))[:,np.newaxis],df.shape[1])
    df = (df-minval)/minmax_val

    df = pd.DataFrame(df,columns=df_cols)

    return df


# # do the cross-validated plots 
# goodclus = clusters[
#     (clusters._av_KSLabels ==2) & 
#     (clusters.winner_model == 'V_spatial') &
#     (clusters.sc_azimuth.notna())
# ]

# tc_g = tc [ 
#     (clusters._av_KSLabels ==2) & 
#     (clusters.winner_model == 'V_spatial') &
#     (clusters.sc_azimuth.notna())
# ]


# %%sort based on the preferred tuning of one anf then look at the other
vis_keys = ['vis_-90','vis_-60', 'vis_-30', 'vis_0', 'vis_30', 'vis_60', 'vis_90']
aud_keys = ['aud_-90', 'aud_-60', 'aud_-30', 'aud_0','aud_30', 'aud_60', 'aud_90']

sorted_g = goodclus.sort_values('sc_azimuth')
# %%
plt.imshow(minmax_norm_df(sorted_g[vis_keys]),aspect = 'auto',cmap='Greys')
# %%
plt.imshow(minmax_norm_df(tc_g[vis_keys].loc[sorted_g.index]),aspect = 'auto',cmap='Greys')
# %%
# perform minmax normalisation for each row 

#
#goodclus = clusters[(clusters._av_KSLabels ==2) &(clusters.vis<0.05) & (clusters.aud>0.05)]
goodclus = clusters[
    (clusters._av_KSLabels ==2) & 
    (clusters.winner_model == 'V_spatial') &
    (clusters.sc_azimuth.notna()) & 
    (clusters.depths_from_sc>-1400) &
    (clusters.depths_from_sc<0)&
    (clusters.ml.notna())

]
sc_azimuths_bins = np.array([-100,-75,-45,-15,15,45,75,100])
# get which bin things belong to 
bin_idx = np.digitize(goodclus.sc_azimuth,bins=sc_azimuths_bins)

vis_tuning_curves = minmax_norm_df(goodclus[vis_keys])
color_ = plt.cm.coolwarm(np.linspace(0,1,sc_azimuths_bins.size+1))



means = [vis_tuning_curves.iloc[bin_idx ==bin_i].mean(axis=0).values for bin_i in np.unique(bin_idx)]
sems = [vis_tuning_curves.iloc[bin_idx ==bin_i].std(axis=0).values/vis_tuning_curves.iloc[bin_idx ==bin_i].shape[0] for bin_i in np.unique(bin_idx)]
# %%
plt.rcParams.update({'font.family':'Verdana'})
plt.rcParams.update({'font.size':14})
fig,ax = plt.subplots(1,1,figsize=(8,5))
xcoords = [-90,-60,-30,0,30,60,90]
[ax.plot(xcoords,means[i],color = color_[bin_i],lw=1) for i,bin_i in enumerate(np.unique(bin_idx))]
#[ax.plot(xcoords,means[i],'.',color = color_[bin_i],markersize=14) for i,bin_i in enumerate(np.unique(bin_idx))]

[ax.fill_between(xcoords,means[i]-sems[i],means[i]+sems[i],color = color_[bin_i],alpha=.5) for i,bin_i in enumerate(np.unique(bin_idx))]

from utils.plotting import off_topspines
off_topspines(ax)
ax.set_xticks(xcoords)
ax.set_xlabel('visual azimuth' + '(' + u'\xb0' + ')')
ax.set_ylabel('rel. response')
ax.set_ylim([0,1])

# %%
hemisphere = np.sign(goodclus.sc_azimuth)
ap_hemi = goodclus.ap*hemisphere
ap_bins= [-9450,-9250,-8800,8800,9250,9450]
bin_idx = np.digitize(ap_hemi,bins=ap_bins)
color_ = plt.cm.coolwarm(np.linspace(0,1,5))
color_[2,0]=212/255
color_[2,1]=42/255
color_[2,2]=1

# %%
means = [vis_tuning_curves.iloc[bin_idx ==bin_i].mean(axis=0).values for bin_i in np.unique(bin_idx)]
sems = [vis_tuning_curves.iloc[bin_idx ==bin_i].std(axis=0).values/vis_tuning_curves.iloc[bin_idx ==bin_i].shape[0] for bin_i in np.unique(bin_idx)]

plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':36})
fig,ax = plt.subplots(1,1,figsize=(8,5))
xcoords = [-90,-60,-30,0,30,60,90]
[ax.plot(xcoords,means[i],color = color_[bin_i-1],lw=1) for i,bin_i in enumerate(np.unique(bin_idx))]
#[ax.plot(xcoords,means[i],'.',color = color_[bin_i],markersize=14) for i,bin_i in enumerate(np.unique(bin_idx))]

[ax.fill_between(xcoords,means[i]-sems[i],means[i]+sems[i],color = color_[bin_i-1],alpha=.5) for i,bin_i in enumerate(np.unique(bin_idx))]

from utils.plotting import off_topspines
off_topspines(ax)
ax.set_xticks(xcoords)
ax.set_xlabel('auditory azimuth' + '(' + u'\xb0' + ')')
ax.set_ylabel('rel. response')
ax.set_ylim([0,1])

plt.savefig("C:\\Users\\Flora\\Pictures\\azimuthal_tuning_aud.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
