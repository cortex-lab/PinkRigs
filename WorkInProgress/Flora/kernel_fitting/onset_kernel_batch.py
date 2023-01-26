# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\Audiovisual") 
from utils.data_manager import get_data_bunch
from pathlib import Path


save_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
dataset = 'naive'
save_path = save_path / dataset / 'kernel_model'
save_path.mkdir(parents=True,exist_ok=True)


recordings = get_data_bunch(dataset)

from src.kernel_regression import kernel_model
kernels = kernel_model(t_bin=0.005,smoothing=0.025)
#nrn_list = [50,140]
kernel_params = {
    'expDef': 'all', 
    't_support_stim':[-0.05,0.6],    
    'rt_params':{'rt_min': None, 'rt_max': None},
    'event_types': ['aud','vis','baseline','motionEnergy'],
    'contrasts': [1],
    'spls': [0.02,0.1],
    'vis_azimuths': [-90,-60,-30,0,30,60,90],
    'aud_azimuths': [-90,-60,-30,0,30,60,90],
    'digitise_cam': False,
    'zscore_cam': 'mad'
    }


for _,rec_info in recordings.iterrows():
    print('Now attempting to fit %s %s, expNum = %.0f, %s' % tuple(rec_info))
    kernels.load_and_format_data(**kernel_params,**rec_info)
    kernels.fit(method='Ridge',ridge_alpha=1,tune_hyper_parameter=False,rank=10,rr_regulariser=0)
    variance_explained = kernels.evaluate(kernel_selection = 'stimgroups',sig_metric = ['explained-variance','explained-variance-temporal'])
    variance_explained.to_csv((save_path / ('%s_%s_%.0f_%s_2022-10-30.csv' % tuple(rec_info))))
    # save variance explained for now



# %%


# %%
