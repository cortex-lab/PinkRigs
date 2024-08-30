import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data, Bunch
from Analysis.pyutils.ev_dat import filter_active_trials


def load_params(paramset='choice'):
    """_summary_

    Args:
        paramset (str, optional): _description_. Defaults to 'choice'.

    Returns:
        _type_: _description_
    """
    if paramset == 'choice':

        timing_params = {
            'onset_time':'timeline_choiceMoveOn',
            'pre_time':0.15,
            'post_time':0
        }

    elif paramset == 'prestim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0.15,
            'post_time':0
        }
    
    elif paramset == 'poststim':
        timing_params = {
            'onset_time':'timeline_audPeriodOn',
            'pre_time':0,
            'post_time':0.15
        }
    
    return timing_params


def load_rec_df(brain_area = 'SCm',paramset='choice',expList=None):
    """function to load the data..."""
    savepath= Path(r'D:\ChoiceEncoding\selected_sessions_%s_%s.csv' % (brain_area,paramset))
    
    ephys_dict = {'spikes':'all','clusters':'all'}
    current_params = {
        'data_name_dict':{'probe0':ephys_dict,'probe1':ephys_dict,
                             'events': {'_av_trials': 'table'},
                             'eyeCam':{'camera':['times','ROIMotionEnergy','_av_motionPCs']},
                            'sideCam':{'camera':['times','ROIMotionEnergy','_av_motionPCs']}},
        'expDef':'multiSpaceWorld',
        'checkEvents':'1',
        'checkSpikes':'1',
        'unwrap_probes':False,
        'cam_hierarchy':None,#['sideCam','eyeCam'],
        'merge_probes':True,
        'region_selection':{'region_name':brain_area,
                            'framework':'Beryl',
                            'min_fraction':30,
                            'goodOnly':True,
                            'min_spike_num':300}
                            
    }


    if expList is None:
        recordings = load_data(
            subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],
            expDate = 'postImplant', 
            **current_params
        )

        p = load_params(paramset=paramset)
        rt_params = {'rt_min':p['post_time']+0.03,'rt_max':1.5}


        exclude_premature_wheel = False
        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]


        (recordings[['subject','expDate','expNum']]).to_csv(savepath)

    else:
        # if type(expList)=str
        #     expList = pd.read_csv(savepath)

        recordings = [load_data(subject = rec.subject,
                                expDate = rec.expDate,
                                expNum = rec.expNum,
                                **current_params) for _,rec in expList.iterrows()]
        recordings = pd.concat(recordings)


        p = load_params(paramset=paramset)
        rt_params = {'rt_min':p['post_time']+0.03,'rt_max':1.5}


        exclude_premature_wheel = False
        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]

    return recordings




