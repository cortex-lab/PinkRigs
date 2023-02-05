# this is the anatomy figure
# general loading functions
# %%
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np


from Analysis.pyutils.batch_data import get_data_bunch
dat_keys = get_data_bunch('naive-allen')

rerun_sig_test=False 
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
from Admin.csv_queryExp import load_ephys_independent_probes
from Processing.pyhist.helpers.util import add_gauss_to_mlapdv

allen_pos_mlapdv, loc_name,curated_label = [],[],[]
is_aud_sig, is_vis_sig = [],[]

for _,session in dat_keys.iterrows():
    r = load_ephys_independent_probes(ephys_dict={'clusters':'all'},**session)
    allen_pos_mlapdv.append(r.iloc[0].probe.clusters.mlapdv)
    loc_name.append(r.iloc[0].probe.clusters.brainLocationAcronyms_ccf_2017)
    curated_label.append(r.iloc[0].probe.clusters._av_KSLabels)
    
    interim_data_sess = interim_data_folder / ('%s/%s/%.0f/%s/sig_test' % tuple(session))
    interim_data_sess.mkdir(parents=True,exist_ok=True)
    interim_data_sess = interim_data_sess / ('%s_%s_%.0f_%s_maxtest.csv' % tuple(session))
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
    is_aud_sig.append(is_signifiant_per_cond[aud_keys].any(axis=1).to_numpy())
    is_vis_sig.append(is_signifiant_per_cond[vis_keys].any(axis=1).to_numpy())

allen_pos_mlapdv,loc_name,curated_label = np.concatenate(allen_pos_mlapdv),np.concatenate(loc_name),np.concatenate(curated_label)
is_aud_sig, is_vis_sig = np.concatenate(is_aud_sig),np.concatenate(is_vis_sig)

allen_pos_mlapdv = add_gauss_to_mlapdv(allen_pos_mlapdv,ml=80,ap=80,dv=0)




is_both = is_aud_sig & is_vis_sig
is_neither = ~is_aud_sig & ~is_vis_sig
is_aud= is_aud_sig & ~is_vis_sig
is_vis= ~is_aud_sig & is_vis_sig
is_good = curated_label==2
is_SC = ['SC'in loc for loc in loc_name]


# %%
from brainrender import Scene
from brainrender.actors import Points

import numpy as np

# Add brain regions
scene = Scene(title="SC aud and vis units", inset=False,root=False)
scene.add_brain_region("SCs",alpha=0.3)
sc = scene.add_brain_region("SCm",alpha=0.3)

allen_pos_dvmlap = allen_pos_mlapdv[:,[1,2,0]]
scene.add(Points(allen_pos_dvmlap[is_both & is_good & is_SC,:], colors='g', radius=30, alpha=0.8))
scene.add(Points(allen_pos_dvmlap[is_vis & is_good & is_SC,:], colors='b', radius=30, alpha=0.8))
scene.add(Points(allen_pos_dvmlap[is_aud & is_good & is_SC,:], colors='m', radius=30, alpha=0.8))
scene.add(Points(allen_pos_dvmlap[is_neither & is_good & is_SC,:], colors='k', radius=15, alpha=0.2))
# plot the neurons in allen atalas space

scene.content
scene.render()

# %%

