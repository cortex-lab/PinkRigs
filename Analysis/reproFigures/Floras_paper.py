# %%
# general loading functions
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

# Figure 1B - example visual neuron
from Admin.csv_queryExp import load_data
recordings = load_data(
    subject = 'FT009',
    expDate = '2021-01-20',
    expNum = 2,
    data_name_dict={
        'events':{'_av_trials':'table'},
    }
    )




# %%
