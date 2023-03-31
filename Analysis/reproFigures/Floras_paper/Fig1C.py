# this is the anatomy figure
# general loading functions
# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.plotting import off_axes,off_topspines
dat_type = 'naive-chronic'
dat_keys = get_data_bunch(dat_type)

from Admin.csv_queryExp import queryCSV

# dat_type = 'AV028_all'
# recordings = queryCSV(subject='AV028',expDef='postactive')

# dat_keys = recordings[['subject','expDate','expNum']]
# dat_keys['probe']='probe0'

#  %%
rerun_sig_test= False 
recompute_csv = False 
recompute_pos_model = False 

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
from Analysis.neural.utils.data_manager import load_cluster_info
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning

# load the ap pref azimuth model 
modelpath = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\anatomy')
modelpath = modelpath / 'aphemi_preferred_azimuth.pickle'
if modelpath.is_file():
    openpickle = open(modelpath,'rb')
    pos_azimuth_fun = pickle.load(openpickle)
else: 
    print('position to azimuth mapping does not exist.')


# %%
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': None, # None means I select the max
    'subselect_neurons':None,
}
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type 
csv_path.mkdir(parents=True,exist_ok=True)
csv_path = csv_path / 'summary_data.csv'

tuning_types = ['vis','aud']
if not csv_path.is_file() or recompute_csv:
    all_dfs = []
    for _,session in dat_keys.iterrows():
    # get generic info on clusters 
        clusInfo = load_cluster_info(**session)
        # add clus_new_columns to clusters
        # significance of responsivity  
        #interim_data_sess = interim_data_folder / ('%s/%s/%.0f/%s/sig_test' % tuple(session))
        interim_data_sess = interim_data_folder / ('%s/%s/%s/%s/sig_test' % tuple(session))
        interim_data_sess.mkdir(parents=True,exist_ok=True)
        #interim_data_sess = interim_data_sess / ('%s_%s_%.0f_%s_maxtest.csv' % tuple(session))
        interim_data_sess = interim_data_sess / ('%s_%s_%s_%s_maxtest.csv' % tuple(session))
        # get significance
        if rerun_sig_test or not interim_data_sess.is_file():
            print('running sig test for %s' % interim_data_sess.__str__())
            from Analysis.neural.src.maxtest import maxtest
            sig_test = maxtest()
            sig_test.load_and_format_data(**session)
            p=sig_test.run(
                n_shuffles=2000,
                savepath= interim_data_sess
            ) # still, rather slow
        else: 
            p = pd.read_csv(interim_data_sess)
        # for each max test get neurons that pass threshold
        bonferroni_p_thr = 0.01/p.columns.size
        is_signifiant_per_cond = p<bonferroni_p_thr
        aud_keys = [k for k in p.keys() if 'aud' in k]
        vis_keys = [k for k in p.keys() if 'vis' in k]

        clusInfo['is_aud_sig']= is_signifiant_per_cond[aud_keys].any(axis=1).to_numpy()
        clusInfo['is_vis_sig']=is_signifiant_per_cond[vis_keys].any(axis=1).to_numpy()
        clusInfo['is_both'] = clusInfo.is_aud_sig & clusInfo.is_vis_sig
        clusInfo['is_neither'] = ~clusInfo.is_aud_sig & ~clusInfo.is_vis_sig
        clusInfo['is_aud']= clusInfo.is_aud_sig & ~clusInfo.is_vis_sig
        clusInfo['is_vis']= ~clusInfo.is_aud_sig & clusInfo.is_vis_sig

        # predict preferred spatial tuning based on position
        clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position
        
        azimuth_pref_estimate = pos_azimuth_fun.predict(clusInfo.aphemi.values.reshape(-1,1))
        
        #get spatial tuning properties
        azi = azimuthal_tuning(session)
        for t in tuning_types:
            tuning_curve_params['which'] = t
            azi.get_rasters_perAzi(**tuning_curve_params)
            clusInfo['is_%s_spatial' % t],clusInfo['%s_preferred_tuning' % t] = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)
            clusInfo['%s_selectivity'% t] = azi.selectivity
            tcs=azi.get_tuning_curves(cv_split=2,azimuth_shuffle_seed=None)
            clusInfo = pd.concat((clusInfo,tcs[tcs.cv_number==1].iloc[:,:-2]),axis=1) # concatenate the actual tuning curves, test set_only



       # then calculate enhancement index at "preferred azimuths". 
        azimuth_pref_estimate = np.digitize(azimuth_pref_estimate,bins=azi.aud.azimuths.values+15)
        azimuth_pref_estimate = azi.aud.azimuths.values[azimuth_pref_estimate]
        clusInfo['enhancement_index'] = azi.get_enhancement_index(at_azimuth=azimuth_pref_estimate)

        
        clusInfo['is_good'] = clusInfo._av_KSLabels==2
        clusInfo['is_SC'] = ['SC'in loc for loc in clusInfo.brainLocationAcronyms_ccf_2017]

        all_dfs.append(clusInfo)
    
    clusInfo = pd.concat(all_dfs,axis=0)    
    if csv_path.is_file():
        # save previous
        old = pd.read_csv(csv_path)
        time_created = datetime.datetime.fromtimestamp(
            csv_path.stat().st_ctime
            ).strftime("%Y-%m-%d-%H%M")
        old_save_path = csv_path.parent / ('summary_data%s.csv' % time_created)
        old.to_csv(old_save_path)

    clusInfo.to_csv(csv_path)
else:
    clusInfo = pd.read_csv(csv_path)

# %%
from Analysis.pyutils.plotting import off_exceptx
sc_clus = clusInfo[clusInfo.is_SC & clusInfo.is_good]
ei = sc_clus.enhancement_index.values
_,ax = plt.subplots(1,1,figsize=(3,3))
ax.hist(
    ei[~np.isnan(ei) & ~np.isinf(ei)],
    bins=np.arange(-3,3,0.1)
)
off_exceptx(ax)
ax.set_xlabel('multisensory enhancement index')

# %%
import plotly.express as px
plotted_tc = 'vis'

goodclus = clusInfo[clusInfo['is_%s' % plotted_tc] & clusInfo.is_good & clusInfo['is_%s_spatial' % plotted_tc] & clusInfo.is_SC]

fig = px.scatter(
    goodclus,
    x='aphemi', y='%s_preferred_tuning' % plotted_tc,color = 'expFolder',symbol='probe', 
    hover_data=['expFolder','probe','_av_IDs']
    )
fig.show()


allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# %%
# _,ax = plt.subplots(len(tuning_types),1,figsize=(5,9),sharey=True)

# maps = {}
# for idx,t in enumerate(tuning_types):
#     print(t)
#     goodclus = clusInfo[clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo['is_%s_spatial' % t] & clusInfo.is_SC]
#     namekeys = [c for c in clusInfo.columns if '%s_' % t in c][-7:]
#     print(namekeys)
#     tcs = goodclus.sort_values('%s_preferred_tuning' % t)
#     tcs = tcs[namekeys]

#     tcs_norm = pd.DataFrame.div(pd.DataFrame.subtract(tcs,tcs.min(axis=1),axis='rows'),
#         (tcs.max(axis=1)+tcs.min(axis=1)),axis='rows')                   


#     ax[idx].imshow(tcs_norm,aspect='auto',cmap='PuRd')
#     ax[idx].set_ylim([269,0])
#     off_axes(ax[idx])

#     goodclus['pos_bin_idx'] = np.digitize(goodclus.aphemi,bins=np.arange(-1000,1000,200))
#     unique_bins = np.unique(goodclus.pos_bin_idx)
#     mean_per_pos = [np.mean(goodclus[goodclus.pos_bin_idx==b]['%s_preferred_tuning' % t]) for b in unique_bins]
#     std_per_pos = [np.std(goodclus[goodclus.pos_bin_idx==b]['%s_preferred_tuning' % t]) for b in unique_bins]
#     maps['%s_mean'% t] = mean_per_pos
#     maps['%s_std' % t ] = std_per_pos

# print(len(maps['vis_mean']),len(maps['aud_mean']))
# _,ax = plt.subplots(1,1,figsize=(2,2))
# ax.scatter(maps['vis_mean'],maps['aud_mean'],marker='o',color='lightblue',edgecolors='k')
# off_topspines(ax)
# ax.set_xlabel('preferred visual azimuth')
# ax.set_ylabel('preferred auditory azimuth')

# %%

# learn preferred azimuth based on location 
if recompute_pos_model:
    from sklearn.linear_model import LinearRegression
    nanpos = np.isnan(goodclus.aphemi.values)
    pos_azimuth_fun = LinearRegression().fit(
        goodclus.aphemi.values[~nanpos,np.newaxis],
        goodclus.vis_preferred_tuning.values[~nanpos]
        )
    # save model 
    modelpath = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\anatomy')
    modelpath = modelpath / 'aphemi_preferred_azimuth.pickle'
    pickle.dump(pos_azimuth_fun, open(modelpath, 'wb'))

# %%
import matplotlib.pyplot as plt
from Analysis.pyutils.plotting import rgb_to_hex
azimuths = np.sort(clusInfo.vis_preferred_tuning.unique())
color_ = plt.cm.coolwarm(np.linspace(0,1,azimuths.size))
t = 'aud'
# plt.scatter(clusInfo.ml,clusInfo.ap,c=clusInfo['%s_preferred_tuning'  % t], lw=0.1, cmap='coolwarm')
# plt.colorbar()
color_ = [rgb_to_hex((c[:3]*255).astype('int')) for c in color_]

# %% 
# look at discriminability 
 
# plotting in brainrender 
from brainrender import Scene
from brainrender.actors import Points

import numpy as np

# Add brain regions
scene = Scene(title="SC aud and vis units", inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.05,color='grey')
sc = scene.add_brain_region("SCm",alpha=0.05,color='grey')

# scene.add(Points(allen_pos_apdvml[clusInfo.is_both & clusInfo.is_good & clusInfo.is_SC,:], colors='g', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good & clusInfo.is_SC,:], colors='b', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good & clusInfo.is_SC,:], colors='m', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good & clusInfo.is_SC,:], colors='k', radius=15, alpha=0.2))


#plot the neurons in allen atalas space
scene.add(Points(allen_pos_apdvml[clusInfo.is_both  & clusInfo.is_good,:], colors='g', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good,:], colors='b', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good,:], colors='m', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good,:], colors='k', radius=15, alpha=0.2))


# for azi,c in zip(azimuths,color_):    
#     scene.add(Points(
#         allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & (clusInfo['%s_preferred_tuning'  % t] == azi),:], 
#         colors=c, 
#         radius=30, 
#         alpha=1
#         ))    
# scene.add(Points(allen_pos_apdvml[~clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good ,:], colors='k', radius=15, alpha=0.1))    

scene.content
scene.render()

# %%
# discriminability

# 
