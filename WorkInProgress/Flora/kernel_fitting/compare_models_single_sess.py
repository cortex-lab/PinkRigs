# here we compare models on exactly the same train-test splits 
# %%
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.neural.utils.plotting import off_topspines,off_axes

from Analysis.neural.src.kernel_model import kernel_model
kernels = kernel_model(t_bin=0.005,smoothing=0.025)

from kernel_params import get_params
dat_params,fit_params,eval_params = get_params()
nrn_list = [571]
#nrn_list = [50,140]
kernels.load_and_format_data(
    subject = 'FT011',
    expDate = '2021-03-24', 
    expDef = 'all',
    expNum = 7,
    probe = 'probe0',
    subselect_neurons=None,
    **dat_params
)
# perform fitting of the most complicated model. 
kernels.fit(**fit_params)

# zero the non-linearity kernels, refit and reevaluate? 


# compare cv, train test 

import matplotlib.pyplot as plt

# calculate degree of overfit
plt.figure()
test_var_nl = kernels.fit_results.test_explained_variance.copy()
dev_var_nl = kernels.fit_results.dev_explained_variance.copy()
overfit = (dev_var_nl - test_var_nl)/dev_var_nl

plt.hist(overfit)
plt.title(np.median(overfit))

# %% refit, omitting some predictors. 
full_feature_matrix = kernels.feature_matrix.copy()
all_keys = list(kernels.feature_column_dict.keys())
to_omit = [k for k in all_keys if 'non-linear' in k]
feature_matrix_omitted = full_feature_matrix.copy()
if to_omit: 
    for k in to_omit:    
        feature_matrix_omitted[:, kernels.feature_column_dict[k]] = 0
        kernels.feature_matrix = feature_matrix_omitted
else:
        kernels.feature_matrix = full_feature_matrix   

kernels.fit(**fit_params)
# %%
plt.plot(test_var_nl,kernels.fit_results.test_explained_variance,'o')
plt.xlabel('non-linear')
plt.ylabel('additive')
plt.plot([-.1,.1],[-.1,.1],'k')

# %%
cidx =  np.where(test_var_nl>kernels.fit_results.test_explained_variance)[0]
kernels.predict()

# %%

kernels.plot_prediction(
    nrnID=kernels.clusIDs[cidx][43],
    plot_stim = True, 
    plot_move=False, 
    sep_choice=False,
    plotted_vis_azimuth = np.array([-1000,-90,-60,-30,0,30,60,90]),
    plotted_aud_azimuth = np.array([-1000,-90,-60,-30,0,30,60,90]),
    plot_train =True,
    plot_test = False,
    plot_pred_train = False,
    plot_pred_test = True,
    )

# %%
