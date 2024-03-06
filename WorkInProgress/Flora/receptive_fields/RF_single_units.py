# %%
# this code fits receptive fields of individual units
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Analysis.neural.src.rf_model import rf_model

subject = 'FT010'
expDate = '2021-03-17'
expNum = '9'
probe = 'probe0'
rf = rf_model(subject=subject,expDate=expDate,expNum = expNum,probe=probe)

# rf.raster_kwargs['pre_time'] = 0.06
# rf.raster_kwargs['smoothing'] =0.025
# rf.raster_kwargs['post_time'] = 0.08
# rf.raster_kwargs['baseline_subtract'] = True


rf.get_significant_rfs(
    n_shuffles=1,
    mode='per-neuron',
    selected_ids=None,
    delay_for_vistrig=0.01,
    cv_split = 2)

# %%
# highest results on test set
x = rf.score.values[:,1]
x[np.isnan(x)] = 0
rf.score.neuronID.values[np.argsort(rf.score.values[:,1])]
# %%
# in reality this is way too much to run. 
from pathlib import Path
import matplotlib.pyplot as plt
azi,elevation,s,ss = rf.get_rf_degs_from_fit()

cID = 204
responses = rf.plot_fit(ID=cID,
                    mode='per-neuron',
                    selected_ids=None,
                    delay_for_vistrig=0.01,
                    cv_split = 2)

# save as svg

which_figure = '%s_%s_%s_%s_nID%.0f_azi%.0f_elevation_%.0f' % (subject,expDate,expNum,probe,cID,azi[rf.score.neuronID==cID][0],elevation[rf.score.neuronID==cID][0])
cpath  = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\Single images')
im_name = 'receptive_fields_singleN' + which_figure + '.svg'
savename = cpath / im_name #'outline_brain.svg'

plt.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
#%%


# %%
# fitting procedure
# fit on train 
# calculate VE from test
# repeat by shuffling the xypos labels 

# plotting function 

# 



# %%
