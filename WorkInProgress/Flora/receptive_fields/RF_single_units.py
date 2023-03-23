# %%
# this code fits receptive fields of individual units
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Analysis.neural.src.rf_model import rf_model

rf = rf_model(subject='FT009',expDate='2021-01-20',expNum = '3')

responses = rf.plot_fit(ID='0', 
    mode='per-depth',
    selected_ids=None,
    delay_for_vistrig=0,
    cv_split = 2
    )

# %%

# fitting procedure
# fit on train 
# calculate VE from test
# repeat by shuffling the xypos labels 

# plotting function 

# 



# %%
