# %%
# this code fits receptive fields of individual units
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Analysis.neural.src.rf_model import rf_model

rf = rf_model(subject='FT009',expDate='2021-01-20',expNum = '4')


# rf.raster_kwargs['pre_time'] = 0.06
# rf.raster_kwargs['smoothing'] =0.025
# rf.raster_kwargs['post_time'] = 0.08
# rf.raster_kwargs['baseline_subtract'] = True


# rf.get_significant_rfs(
#     n_shuffles=1,
#     mode='per-depth',
#     selected_ids=None,
#     delay_for_vistrig=0.01,
#     cv_split = 2)
# in reality this is way too much to run. 

responses = rf.plot_fit(ID=82,
                    mode='per-neuron',
                    selected_ids=None,
                    delay_for_vistrig=0.01,
                    cv_split = 2)
# %%
# fitting procedure
# fit on train 
# calculate VE from test
# repeat by shuffling the xypos labels 

# plotting function 

# 



# %%
