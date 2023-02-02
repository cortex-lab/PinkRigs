# this is the anatomy figure
# general loading functions
# %%
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Analysis.pyutils.batch_data import get_data_bunch

dat_keys = get_data_bunch('naive')

rerun_sig_test=False 
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
# %%

from Admin.csv_queryExp import load_ephys_independent_probes


# %%
for _,session in dat_keys.iterrows():
    r = load_ephys_independent_probes(ephys_dict={'spikes':'all','clusters':'all'},**session)
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
    is_aud_sig = is_signifiant_per_cond[aud_keys].any(axis=1).to_numpy()
    is_vis_sig = is_signifiant_per_cond[vis_keys].any(axis=1).to_numpy()

    is_both = is_aud_sig & is_vis_sig
    is_neither = ~is_aud_sig & ~is_vis_sig
    is_aud= is_aud_sig & ~is_vis_sig
    is_vis= ~is_aud_sig & is_vis_sig

    # load cluster location in allen ccf and unit quality metrics



# %%
