# %% 
import sys,shutil
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import numpy as np
import pandas as pd
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json


from Admin.csv_queryExp import queryCSV


subject_set = ['AV025','AV030','AV034']
my_expDef = 'multiSpaceWorld_checker_training'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

recordings = queryCSV(subject = subject_set,expDate='postImplant',expDef=my_expDef,checkEvents='1',checkSpikes='1',unwrap_independent_probes=True)

# %%



fit_tag = 'stimChoice'

recordings = recordings[['subject','expDate','expNum','probe']]

# %%
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag

save_path.mkdir(parents=True,exist_ok=True)

from Analysis.neural.src.kernel_model import kernel_model
from kernel_params import get_params



failed_recs = []
for _,rec_info in recordings.iterrows():

    try: 
        print('Now attempting to fit %s %s, expNum = %s, %s' % tuple(rec_info))

        # since kernels is a class, I think it is safer to recall it after each fit... I think that is why -1000 kept accumulating in dat_params, for example
        kernels = kernel_model(t_bin=0.005,smoothing=0.025)
        dat_params,fit_params,eval_params = get_params()

        kernels.load_and_format_data(**dat_params,**rec_info)
        kernels.fit(**fit_params)
        variance_explained = kernels.evaluate(**eval_params)


        # save all the results
        nametag = '%s_%s_%s_%s' % tuple(rec_info)

        # create a folder 
        curr_save_path  = save_path / nametag
        # remove it if it already existed
        if curr_save_path.is_dir():
            shutil.rmtree(curr_save_path)
        curr_save_path.mkdir(parents=True,exist_ok=True)

        variance_explained.to_csv((curr_save_path / ('variance_explained_batchKernel.csv')))
        my_kernels = kernels.calculate_kernels()

        for k in my_kernels.keys():
            np.save(
                (curr_save_path / ('%s.npy' % k)), 
                my_kernels[k]
            )
        
        np.save(
            (curr_save_path / 'clusIDs.npy'), 
            kernels.clusIDs
        )

    except:
        print('Failed to fit %s %s, expNum = %s, %s' % tuple(rec_info))
        failed_recs.append(rec_info)


# %%
failed_recs = pd.DataFrame(failed_recs,columns = ['subject','expDate','expNum','probe'])

failed_recs.to_csv((save_path / 'failed_to_fit.csv'))    
# save the parameters of fitting
save_dict_to_json(dat_params,save_path / 'dat_params.json')
save_dict_to_json(fit_params,save_path / 'fit_params.json')
save_dict_to_json(eval_params,save_path / 'eval_params.json')




# %%

# %%
