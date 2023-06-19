#%% 
# this func will take all spontaneous and sparse noise recordings 
# such that we can run the anatmap specifically on those recordings 
import sys

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
# ONE loader from the PinkRig Pipeline
from Analysis.pyutils.plotting import plot_anatmaps_longitudinal_

plot_anatmaps_longitudinal_(subject='AV030',probe='probe1',savefigs=True)


# %%
