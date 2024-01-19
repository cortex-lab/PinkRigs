# %%


from shutil import move
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from scipy.stats import median_abs_deviation,zscore
from Analysis.neural.utils.spike_dat import get_binned_rasters
from Analysis.pyutils.video_dat import digitise_motion_energy
import matplotlib.pyplot as plt
from Analysis.pyutils.plotting import off_topspines

from Admin.csv_queryExp import load_data

subject = 'FT008'
expDate = '2021-01-15'
expNum= 5
cam = 'frontCam'
probe = 'probe1'

data_dict = {cam:{'camera':'all','_av_motionPCs':'all'}}
recordings = load_data(subject = subject,expDate= expDate, expNum=expNum,data_name_dict=data_dict)

camera = recordings.iloc[0][cam].camera

# %%
digitise_motion_energy(camera.times,camera.ROIMotionEnergy,plot_sample=True,min_on_time=0.05,min_off_time=0.05)
plt.show()
print('done')
# %%
