# this is the anatomy figure
# general loading functions
# %%
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np

from Analysis.pyutils.batch_data import get_data_bunch
dat_type = 'postactive'
dat_keys = get_data_bunch(dat_type)

rerun_sig_test=False 
recompute_csv = False 

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
from Analysis.neural.utils.data_manager import load_cluster_info
from Processing.pyhist.helpers.util import add_gauss_to_apdvml
from Analysis.neural.src.azimuthal_tuning import azimuthal_tuning,get_discriminability,get_tc_correlations

tuning_curve_params = { 
    'contrast': None, # means I select the max
    'spl': None, # means I select the max
    'subselect_neurons':None,
}

csv_path = interim_data_folder / dat_type 
csv_path.mkdir(parents=True,exist_ok=True)
csv_path = csv_path / 'summary_data.csv'

if not csv_path.is_file() or recompute_csv:
    all_dfs = []
    for _,session in dat_keys.iterrows():
    # get generic info on clusters 
        clusInfo = load_cluster_info(**session)
        # add clus_new_columns to clusters
        # significance of responsivity  
        interim_data_sess = interim_data_folder / ('%s/%s/%.0f/%s/sig_test' % tuple(session))
        #interim_data_sess = interim_data_folder / ('%s/%s/%s/%s/sig_test' % tuple(session))
        interim_data_sess.mkdir(parents=True,exist_ok=True)
        interim_data_sess = interim_data_sess / ('%s_%s_%.0f_%s_maxtest.csv' % tuple(session))
        #interim_data_sess = interim_data_sess / ('%s_%s_%s_%s_maxtest.csv' % tuple(session))
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

        #get spatial tuning properties
        azi = azimuthal_tuning(session)
        tuning_type = 'vis'
        tuning_curve_params['which'] = tuning_type
        # get the visual tuning curves
        azi.get_rasters_perAzi(**tuning_curve_params)
        tc = azi.get_tuning_curves(cv_split=1,azimuth_shuffle_seed=None)
        clusInfo['vis_discriminability']= get_discriminability(tc[tc.cv_number==0])
        clusInfo['vis_preferred_tuning']= tc.preferred_tuning.values.astype('float')

        tuning_type = 'aud'
        tuning_curve_params['which'] = tuning_type
        # get the visual tuning curves
        azi.get_rasters_perAzi(**tuning_curve_params)
        # have to save the output of this step because it seems to be taking forever...
        # and I am gonna kill myself. 
        tc = azi.get_tuning_curves(cv_split=1,azimuth_shuffle_seed=None)
        clusInfo['aud_discriminability']= get_discriminability(tc[tc.cv_number==0])
        clusInfo['aud_preferred_tuning']= tc.preferred_tuning.values.astype('float')

        clusInfo['is_both'] = clusInfo.is_aud_sig & clusInfo.is_vis_sig
        clusInfo['is_neither'] = ~clusInfo.is_aud_sig & ~clusInfo.is_vis_sig
        clusInfo['is_aud']= clusInfo.is_aud_sig & ~clusInfo.is_vis_sig
        clusInfo['is_vis']= ~clusInfo.is_aud_sig & clusInfo.is_vis_sig
        clusInfo['is_good'] = clusInfo._av_KSLabels==2
        clusInfo['is_SC'] = ['SC'in loc for loc in clusInfo.brainLocationAcronyms_ccf_2017]

        all_dfs.append(clusInfo)
    
    clusInfo = pd.concat(all_dfs,axis=0)
    clusInfo['hemi'] = np.sign(clusInfo.ml-5600)
    clusInfo['aphemi'] = (clusInfo.ap-8500)*clusInfo.hemi
    clusInfo.to_csv(csv_path)
else:
    clusInfo = pd.read_csv(csv_path)

# %%
import plotly.express as px

goodclus = clusInfo[clusInfo.is_vis & clusInfo.is_good]

fig = px.scatter(
    goodclus,
    x='aphemi', y='vis_preferred_tuning',
    hover_data=['expFolder','probe']
    )
fig.show()


allen_pos_apdvml = clusInfo[['ap','dv','ml']].values
allen_pos_apdvml= add_gauss_to_apdvml(allen_pos_apdvml,ml=80,ap=80,dv=0)
# %% 
# look at discriminability 
 

# plotting in brainrender 
from brainrender import Scene
from brainrender.actors import Points

import numpy as np

# Add brain regions
scene = Scene(title="SC aud and vis units", inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.3)
sc = scene.add_brain_region("SCm",alpha=0.3)

# scene.add(Points(allen_pos_apdvml[clusInfo.is_both & clusInfo.is_good & clusInfo.is_SC,:], colors='g', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_vis & clusInfo.is_good & clusInfo.is_SC,:], colors='b', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_aud & clusInfo.is_good & clusInfo.is_SC,:], colors='m', radius=30, alpha=0.8))
# scene.add(Points(allen_pos_apdvml[clusInfo.is_neither & clusInfo.is_good & clusInfo.is_SC,:], colors='k', radius=15, alpha=0.2))
# plot the neurons in allen atalas space
scene.add(Points(allen_pos_apdvml[clusInfo.is_both,:], colors='g', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_vis,:], colors='b', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_aud,:], colors='m', radius=30, alpha=0.8))
scene.add(Points(allen_pos_apdvml[clusInfo.is_neither,:], colors='k', radius=15, alpha=0.2))
scene.content
scene.render()

# %%
# discriminability

# 
