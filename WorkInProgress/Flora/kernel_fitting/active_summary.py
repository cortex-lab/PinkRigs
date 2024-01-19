# %%
# utility functions
import sys
import pandas as pd
import numpy as np
# visualisations 
import seaborn as sns
import matplotlib.pyplot as plt

# my specialised functions
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_axes,off_topspines
from Analysis.neural.utils.spike_dat import bombcell_sort_units
from kernel_utils import load_VE_per_cluster



dataset = 'trained-active-curated'
fit_tag = 'additive-fit'
clusInfo = load_VE_per_cluster(dataset,fit_tag)
#bc_class = bombcell_sort_units(clusInfo)



# %%

#sns.pairplot(clusInfo[bc_class=='good'],vars=['kernelVE_aud','kernelVE_move_kernel'],plot_kws=dict(marker="o",alpha=.7))


# %%
