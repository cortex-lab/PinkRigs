"""
kernel model fitting implementation that allows it to also run as arrayjob on the cluster
"""
# %%
import sys
import time
from pathlib import Path
import pandas as pd 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
sys.path.insert(0,r'/lustre/home/zcbtfta/code')

from Analysis.pyutils.io import save_dict_to_json

def train_kernel(rank=1):
    dataset = 'naive-allen'
    fit_tag = 'additive-fit'

    cluster_fit = True

    if cluster_fit:
        target_folder = Path(r'/lustre/home/zcbtfta/data')
        interim_data_folder = Path(r'/lustre/home/zcbtfta/results')
        save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag
        save_path.mkdir(parents=True,exist_ok=True)

    else: 
        target_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\data')

        interim_data_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')
        save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag
        save_path.mkdir(parents=True,exist_ok=True)

    recordings = pd.read_csv(target_folder / 'recordings.csv')

    from Analysis.neural.src.kernel_model import kernel_model
    kernels = kernel_model(t_bin=0.005,smoothing=0.025)
    from kernel_params import get_params
    dat_params,fit_params,eval_params = get_params()


    for idx,rec_info in recordings.iterrows():
    # print('Now attempting to fit %s %s, expNum = %.0f, %s' % tuple(rec_info))
        results_path = (save_path / ('%s_%s_%.0f_%s.csv' % tuple(rec_info[:4])))
        if (i==(rank-1)) and not results_path.is_file(): 
            t0 = time.time()
            df = recordings[idx:idx+1]
            kernels.load_and_format_data(**dat_params,recordings=df,**rec_info[:3])
            kernels.fit(**fit_params)
            variance_explained = kernels.evaluate(**eval_params)
            variance_explained.to_csv(results_path)
            print('time to fit-evaluate:',time.time()-t0,'s')
    # # save the parameters of fitting
    save_dict_to_json(dat_params,save_path / 'dat_params.json')
    save_dict_to_json(fit_params,save_path / 'fit_params.json')
    save_dict_to_json(eval_params,save_path / 'eval_params.json')

# %%
if __name__ == "__main__":  
  train_kernel(rank=sys.argv[1]) 
  #train_kernel(rank='1') 