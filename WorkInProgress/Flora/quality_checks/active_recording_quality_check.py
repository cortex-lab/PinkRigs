# %%
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
import Analysis.neural.utils.data_manager as dat

sc_mice_list = dat.get_sc_mice_list()
savepath = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\all_active_exps.csv')

# %%
#recordings = dat.get_behavior_quality_scores(subject='all',expDate = 'postImplant', savepath=savepath,trim_bad = True)
import pandas as pd
recordings = dat.get_sessions_with_units('extended',subject=['AV024','AV028','FT038','FT039'],expDate = 'postImplant', savepath=None)
#recordings = pd.read_csv(r'C:\Users\Flora\Documents\Processed data\Audiovisual\sc_active_exps.csv')
#days2check,_ = dat.get_highest_yield_unique_ephys(recordings,probe='probe0')
#days2check2,_ = dat.get_highest_yield_unique_ephys(recordings,probe='probe1')

# %% 
# save both days to check and drop duplicates
import pandas as pd
to_process_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\sc_selected_active_exps.csv')
days_to_check_tot = pd.concat((days2check,days2check2))
days_to_check_tot = days_to_check_tot.drop_duplicates(subset=['Subject','expDate','expNum'])
days_to_check_tot.to_csv(to_process_path,index=False)
# %%

# check the receptive field location of these units if they exist
# 
from pathlib import Path
shloc_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\AV008\imec1_SC_shank_pos.npy')
s = np.load(shloc_path)

# %%
