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
dat_type = 'naive-allen'
dat_keys = get_data_bunch(dat_type)

from Admin.csv_queryExp import queryCSV

# dat_type = 'AV028_all'
# recordings = queryCSV(subject='AV028',expDef='postactive')

# dat_keys = recordings[['subject','expDate','expNum']]
# dat_keys['probe']='probe0'

#  %%
rerun_sig_test= False 
recompute_csv = True 
recompute_pos_model = False 

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
from Analysis.neural.utils.data_manager import load_cluster_info
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning
from Analysis.neural.src.movements import movement_correlation


# load the ap pref azimuth model 
modelpath = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\anatomy')
modelpath = modelpath / 'aphemi_preferred_azimuth.pickle'
if modelpath.is_file():
    openpickle = open(modelpath,'rb')
    pos_azimuth_fun = pickle.load(openpickle)
else: 
    print('position to azimuth mapping does not exist.')

m = movement_correlation()
# %%
tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': None, # None means I select the max
    'subselect_neurons':None,
    'trim_type':None, 
    'trim_fraction':None
}
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dat_type 
csv_path.mkdir(parents=True,exist_ok=True)
csv_path = csv_path / 'summary_data.csv'

tuning_types = ['vis','aud']
cv_names = ['train','test']
if not csv_path.is_file() or recompute_csv:
    all_dfs = []
    for _,session in dat_keys.iterrows():
    # get generic info on clusters 
        print(*session)

        ################## MAXTEST #################################
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
        #clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi # calculate relative ap*hemisphre position
        
       # azimuth_pref_estimate = pos_azimuth_fun.predict(clusInfo.aphemi.values.reshape(-1,1))
        
        ################ MOVEMENT ########################
        clusInfo['movement_correlation'],clusInfo['is_movement_correlated'] = m.session_permutation(session,0.05)

        ################ SPATIAL TUNING ######################
        azi = azimuthal_tuning(session)
        for t in tuning_types:
            tuning_curve_params['which'] = t
            azi.get_rasters_perAzi(**tuning_curve_params)
            tcs,is_selective = azi.get_significant_fits(curve_type= 'gaussian',metric='svd')

            # old methohen using the rossi et al. selectivity method 
            #clusInfo['is_%s_spatial' % t],clusInfo['%s_preferred_tuning' % t] = azi.calculate_significant_selectivity(n_shuffles=100,p_threshold=0.05)
            #clusInfo['%s_selectivity'% t] = azi.selectivity
            #tcs=azi.get_tuning_curves(cv_split=2,azimuth_shuffle_seed=None) 
            clusInfo['is_%s_spatial' % t]  = is_selective       
            for i,cv in enumerate(cv_names):
                clusInfo = pd.concat((clusInfo,tcs[tcs.cv_number==i].add_suffix('_'+ cv)),axis=1) # concatenate the actual tuning curves, test set_only

            clusInfo = pd.concat((azi.tc_params.add_suffix(t),clusInfo),axis=1)


       # then calculate enhancement index at "preferred azimuths". 
        # azimuth_pref_estimate = np.digitize(azimuth_pref_estimate,bins=azi.aud.azimuths.values+15)
        # azimuth_pref_estimate = azi.aud.azimuths.values[azimuth_pref_estimate]

        azimuth_pref_estimate_aud = np.digitize(clusInfo.x0aud.values,bins=np.arange(-75,105,30),right=False)
        azimuth_pref_estimate_aud = azi.aud.azimuths.values[azimuth_pref_estimate_aud]
        azimuth_pref_estimate_aud[azimuth_pref_estimate_aud==0] = np.nan

        azimuth_pref_estimate_vis = np.digitize(clusInfo.x0vis.values,bins=np.arange(-75,105,30),right=False)
        azimuth_pref_estimate_vis = azi.aud.azimuths.values[azimuth_pref_estimate_vis]
        azimuth_pref_estimate_vis[azimuth_pref_estimate_vis==0] = np.nan

        at_azimuth_values = np.concatenate((azimuth_pref_estimate_vis[:,np.newaxis],azimuth_pref_estimate_aud[:,np.newaxis]),axis=1)
        clusInfo['enhancement_index_pref'] = azi.get_enhancement_index_per_nrn(at_azimuth_values)    

        at_azimuth_values = np.concatenate((azimuth_pref_estimate_vis[:,np.newaxis],azimuth_pref_estimate_aud[:,np.newaxis]*-1),axis=1)
        clusInfo['enhancement_index_antipref,aud'] = azi.get_enhancement_index_per_nrn(at_azimuth_values)

        at_azimuth_values = np.concatenate((azimuth_pref_estimate_vis[:,np.newaxis]*-1,azimuth_pref_estimate_aud[:,np.newaxis]),axis=1)
        clusInfo['enhancement_index_antipref,vis'] = azi.get_enhancement_index_per_nrn(at_azimuth_values)

        ################### KERNEL FIT RESULTS #############################
        foldertag = r'kernel_model\additive-fit'
        csvname = '%s_%s_%.0f_%s.csv' % tuple(session)
        kernel_fit_results = interim_data_folder / dat_type  / foldertag / csvname

        kernel_events_to_save = ['aud', 'baseline', 'motionEnergy', 'vis']
        for k in kernel_events_to_save:
            tag = 'kernelVE_%s' % k 
            if kernel_fit_results.is_file():
                kernel_fits = pd.read_csv(kernel_fit_results)
                kernel_events_to_save  = np.unique(kernel_fits.event)
                # match neurons 
                curr_set = kernel_fits[(kernel_fits.event==k) & (kernel_fits.cv_number==1)]             

                # concatenate with clusInfo
                unmatched_clus_idx = np.setdiff1d(clusInfo._av_IDs,curr_set.clusID)
                if len(unmatched_clus_idx)==0:
                    clusInfo[tag] = curr_set.VE.values
                else:
                    VEs = curr_set.VE.values
                    newVE = []
                    matched_clusIDs = curr_set.clusID
                    for c in clusInfo._av_IDs:
                        idx = np.where(matched_clusIDs==c)[0]
                        if len(idx)==1:
                            newVE.append(VEs[idx[0]])
                        else:
                            newVE.append(np.nan)  
                    
                    clusInfo[tag] = newVE                     

            else: 
                clusInfo[tag] = np.nan


        #################### MISC ###########################################
        clusInfo['is_good'] = clusInfo._av_KSLabels==2

        clusInfo['is_SC'] = ['SC'in loc for loc in clusInfo.brainLocationAcronyms_ccf_2017]

        all_dfs.append(clusInfo)
    
    # temproary hack 
    #all_dfs = [d.drop(columns=['sc_azimuth', 'sc_elevation', 'sc_surface']) if 'sc_azimuth' in d.columns else d for d in all_dfs]
    clusInfo = pd.concat(all_dfs,axis=0)    

    clusInfo['vis_score_test']  =  clusInfo.score_test.iloc[:,0]
    clusInfo['aud_score_test']  =  clusInfo.score_test.iloc[:,1]

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
sc_clus = clusInfo[clusInfo.is_SC & clusInfo.is_good & clusInfo.is_aud_spatial & clusInfo.is_vis_spatial]
ei = sc_clus['enhancement_index_antipref,aud'].values
_,ax = plt.subplots(1,1,figsize=(3,3))
ax.hist(
    ei[~np.isnan(ei) & ~np.isinf(ei)],
    bins=np.arange(-2,2,0.2)
)
off_exceptx(ax)
ax.set_xlabel('multisensory enhancement index')

# %%

# we consider units spatual that 
# clusInfo['is_vis_spatial'] = clusInfo.vis_score_test>0.2
# clusInfo['is_aud_spatial'] = clusInfo.aud_score_test>0.2
# %%
# import plotly.express as px
# plotted_tc = 'vis'

# goodclus = clusInfo[clusInfo['is_%s' % plotted_tc] & clusInfo.is_good & clusInfo['is_%s_spatial' % plotted_tc] & clusInfo.is_SC]

# fig = px.scatter(
#     goodclus,
#     x='aphemi', y='%s_preferred_tuning' % plotted_tc,color = 'expFolder',symbol='probe', 
#     hover_data=['expFolder','probe','_av_IDs']
#     )
# fig.show()


allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# %%
_,ax = plt.subplots(len(tuning_types),1,figsize=(5,9),sharey=True)

maps = {}
for idx,t in enumerate(tuning_types):
    print(t)
    goodclus = clusInfo[clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo['is_%s_spatial' % t] & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])
]
    namekeys = [c for c in clusInfo.columns if ('%s_' % t in c) & ('_train' in c)][:7]
    print(namekeys)
    tcs = goodclus.sort_values('x0%s' % t)
    tcs = tcs[namekeys]

    tcs_norm = pd.DataFrame.div(pd.DataFrame.subtract(tcs,tcs.min(axis=1),axis='rows'),
        (tcs.max(axis=1)+tcs.min(axis=1)),axis='rows')                   


    ax[idx].matshow(tcs_norm,aspect='auto',cmap='PuRd')
    ax[idx].set_ylim([240,0])
    off_axes(ax[idx])
# 
    goodclus['pos_bin_idx'] = np.digitize(goodclus.aphemi,bins=np.arange(-1000,1000,250))
    unique_bins = np.unique(goodclus.pos_bin_idx)
    mean_per_pos = [np.mean(goodclus[goodclus.pos_bin_idx==b]['x0%s' % t]) for b in unique_bins]
    std_per_pos = [np.std(goodclus[goodclus.pos_bin_idx==b]['x0%s' % t]) for b in unique_bins]
    maps['%s_mean'% t] = mean_per_pos
    maps['%s_std' % t ] = std_per_pos

print(len(maps['vis_mean']),len(maps['aud_mean']))
_,ax = plt.subplots(1,1,figsize=(2,2))
ax.scatter(maps['vis_mean'],maps['aud_mean'],marker='o',color='lightblue',edgecolors='k')
off_topspines(ax)
ax.plot([-90,90],[-90,90],'k--',alpha=0.3)

ax.set_xlim([-90,90])
ax.set_ylim([-90,90])
ax.set_xlabel('preferred visual azimuth')
ax.set_ylabel('preferred auditory azimuth')

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
#azimuths = np.sort(clusInfo.vis_preferred_tuning.unique())
color_ = plt.cm.coolwarm(np.linspace(0,1,7))
t = 'aud'
# plt.scatter(clusInfo.ml,clusInfo.ap,c=clusInfo['%s_preferred_tuning'  % t], lw=0.1, cmap='coolwarm')
# plt.colorbar()
color_ = [rgb_to_hex((c[:3]*255).astype('int')) for c in color_]

# %% 
# look at discriminability 
 
# plotting in brainrender 
import brainrender as br
import numpy as np

# Add brain regions
scene = br.Scene(title="SC aud and vis units", inset=False,root=False)

scene.add_brain_region("SCs",alpha=0.07,color='sienna')
#sc = scene.add_brain_region("SCm",alpha=0.04,color='teal')
# scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good & clusInfo.is_SC,:], colors='grey', radius=20, alpha=0.7))
# scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good & clusInfo.is_SC,:], colors='b', radius=20, alpha=0.7))
# scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good & clusInfo.is_SC,:], colors='m', radius=20, alpha=0.7))
# scene.add(br.actors.Points(allen_pos_apdvml[clusInfo.is_both & clusInfo.is_good & clusInfo.is_SC,:], colors='g', radius=20, alpha=0.7))

#plot the neurons in allen atalas space
# scene.add(Points(allen_pos_apdvml[clusInfo.is_both  & clusInfo.is_good,:], colors='g', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good,:], colors='b', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good,:], colors='m', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good,:], colors='k', radius=15, alpha=0.2))

# for azi,c in zip(azimuths,color_):    
#     scene.add(Points(
#         allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & (clusInfo['%s'  % t] == azi),:], 
#         colors=c, 
#         radius=30, 
#         alpha=1
#         ))    
# scene.add(Points(allen_pos_apdvml[~clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good ,:], colors='k', radius=15, alpha=0.1))    
t = 'vis'
from Analysis.pyutils.plotting import brainrender_scattermap

dots_to_plot = allen_pos_apdvml[clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t]),:]
dot_colors = brainrender_scattermap(clusInfo['x0%s' % t][clusInfo['is_%s_spatial'% t] & clusInfo['is_%s' % t] & clusInfo.is_good & clusInfo.is_SC & ~np.isnan(clusInfo['x0%s' % t])],vmin = -90,vmax=90,n_bins=15,cmap='coolwarm')

scene.add(br.actors.Points(dots_to_plot, colors=dot_colors, radius=30, alpha=0.5))


scene.render()

# %%
# discriminability

# 
