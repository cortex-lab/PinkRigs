# %% 
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json


from pathlib import Path 
from Admin.csv_queryExp import queryCSV


subject_set = 'AV025'
my_expDef = 'postactive'
recordings = queryCSV(subject = subject_set,expDate='postImplant',expDef=my_expDef)

# %%
probe = 'probe0'
recordings = recordings[(recordings.extractEvents=='1') & (recordings.extractSpikes=='1,1')]
recordings  = recordings[['subject','expDate','expNum']]
recordings['probe'] = probe


recordings = recordings.iloc[1:]
dataset = subject_set + my_expDef + probe

fit_tag = 'stimChoice'


# %%
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag

save_path.mkdir(parents=True,exist_ok=True)

from Analysis.neural.src.kernel_model import kernel_model

kernels = kernel_model(t_bin=0.005,smoothing=0.025)
from kernel_params import get_params
dat_params,fit_params,eval_params = get_params()


for _,rec_info in recordings.iterrows():


    print('Now attempting to fit %s %s, expNum = %s, %s' % tuple(rec_info))
    kernels.load_and_format_data(**dat_params,**rec_info)
    kernels.fit(**fit_params)
    variance_explained = kernels.evaluate(**eval_params)
    variance_explained.to_csv((save_path / ('%s_%s_%s_%s.csv' % tuple(rec_info))))
    print('probably video did not exist.')
    
# save the parameters of fitting
save_dict_to_json(dat_params,save_path / 'dat_params.json')
save_dict_to_json(fit_params,save_path / 'fit_params.json')
save_dict_to_json(eval_params,save_path / 'eval_params.json')


# %%
