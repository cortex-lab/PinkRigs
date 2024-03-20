import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data
from Analysis.pyutils.ev_dat import filter_active_trials


# ,'AV020','AV025','AV030','AV034'
def load_rec_df(recompute=False):
    """function to load the data..."""
    savepath= Path(r'D:\ChoiceEncoding\neural_choice.csv')
    if recompute:
        ephys_dict = {'spikes':'all','clusters':'all'}
        recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,
                                                 'events': {'_av_trials': 'table'},
                                                 'eyeCam':{'camera':['times','ROIMotionEnergy']},
                                                 'sideCam':{'camera':['times','ROIMotionEnergy']}},
                                subject = ['FT030','FT031','AV005','AV008','AV014'],expDate='postImplant',
                                expDef='multiSpaceWorld',
                                checkEvents='1',
                                checkSpikes='1',
                                unwrap_probes=False, merge_probes=True,
                                region_selection={'region_name':'SC','min_fraction':25,'goodOnly':True,'min_spike_num':300})

        # can toughen this critera...
        rt_params = {'rt_min':0.03,'rt_max':1.5}
        exclude_premature_wheel = False
        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>250]

        recordings.to_csv(savepath)

    else:
        recordings = pd.read_csv(savepath)

    return recordings


def load_params(paramset='choice'):
    if paramset=='choice':

        timing_params = {
            'onset_time':'timeline_choiceMoveOn',
            'pre_time':0.15,
            'post_time':0
        }

    elif paramset=='prestim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0.15,
            'post_time':0
        }
    
    elif paramset=='poststim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0,
            'post_time':0.15
        }
    
    return timing_params