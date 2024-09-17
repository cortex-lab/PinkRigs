import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import numpy as np
import pandas as pd
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json  
dataset = 'naive-allen'
fit_tag = 'additive-nl-comp'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag

save_path.mkdir(parents=True,exist_ok=True)

recordings = get_data_bunch(dataset)

from Analysis.neural.src.kernel_model import kernel_model

kernels = kernel_model(t_bin=0.005,smoothing=0.025)
from kernel_params import get_params
dat_params,fit_params,eval_params = get_params()


for _,rec_info in recordings.iterrows():
    print('Now attempting to fit %s %s, expNum = %.0f, %s' % tuple(rec_info))
    try:
        kernels.load_and_format_data(**dat_params,**rec_info)

        kernels.fit(**fit_params)

        # zero the non-linearity kernels, refit and reevaluate? 

        test_var_nl = kernels.fit_results.test_explained_variance.copy()
        dev_var_nl = kernels.fit_results.dev_explained_variance.copy()
        overfit = (dev_var_nl - test_var_nl)/dev_var_nl


        # % refit, omitting some predictors. 
        full_feature_matrix = kernels.feature_matrix.copy()
        all_keys = list(kernels.feature_column_dict.keys())
        to_omit = [k for k in all_keys if 'non-linear' in k]
        feature_matrix_omitted = full_feature_matrix.copy()
        print(to_omit)
        if to_omit: 
            for k in to_omit:    
                feature_matrix_omitted[:, kernels.feature_column_dict[k]] = 0
                kernels.feature_matrix = feature_matrix_omitted
        else:
                kernels.feature_matrix = full_feature_matrix   

        kernels.fit(**fit_params)

        d= np.array([test_var_nl,kernels.fit_results.test_explained_variance])
        df = pd.DataFrame(d.T,columns=['additive','gain'])

        df.to_csv((save_path / ('%s_%s_%.0f_%s.csv' % tuple(rec_info))))
    except:
        print('probably video did not exist.')
    
    
# save the parameters of fitting
save_dict_to_json(dat_params,save_path / 'dat_params.json')
save_dict_to_json(fit_params,save_path / 'fit_params.json')
save_dict_to_json(eval_params,save_path / 'eval_params.json')
