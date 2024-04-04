import sys
from pathlib import Path
import pandas as pd 
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.ev_dat import filter_active_trials


def load_for_movement_correlation(dataset='naive',recompute_data_selection=False):
    """function to load the data..."""

    current_params = {
        'data_name_dict':{ 'events': {'_av_trials': 'table'}, 
                           'frontCam':{'camera':['times','ROIMotionEnergy']},
                           'eyeCam':{'camera':['times','ROIMotionEnergy']},
                           'sideCam':{'camera':['times','ROIMotionEnergy']}},
    }
    savepath = r'D:\VideoAnalysis\%s_dataset.csv' % dataset

    
    if (dataset == 'postactive') & recompute_data_selection:
        recordings = load_data(subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],
                               expDef = 'postactive',
                                checkEvents='1',
                                cam_hierarchy= ['sideCam','eyeCam','frontCam'],
                                **current_params)

    if (dataset == 'active') & recompute_data_selection:
        recordings = load_data(subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],
                               checkEvents='1',expDate='postImplant',
                               expDef = 'multiSpaceWorld',
                                cam_hierarchy= ['sideCam','eyeCam','frontCam'],
                                **current_params)  

        exclude_premature_wheel = False
        rt_params = {'rt_min':0.03,'rt_max':1.5}

        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]  


    if (dataset == 'activeWithSpike') & recompute_data_selection:
        # get the spikes into the data name dict
        ephys_dict = {'spikes':'all','clusters':'all'}
        current_params = {
        'data_name_dict':{'probe0':ephys_dict,'probe1':ephys_dict,
                          'events': {'_av_trials': 'table'}, 
                           'frontCam':{'camera':['times','ROIMotionEnergy']},
                           'eyeCam':{'camera':['times','ROIMotionEnergy']},
                           'sideCam':{'camera':['times','ROIMotionEnergy']}}
        }     

        recordings = load_data(subject = ['AV008'],
                               checkEvents='1',expDate='postImplant',
                               expDef = 'multiSpaceWorld',
                               checkSpikes='1', merge_probes=True,
                                cam_hierarchy= ['sideCam','eyeCam','frontCam'],
                                **current_params) 
         

        exclude_premature_wheel = False
        rt_params = {'rt_min':0.03,'rt_max':1.5}

        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]   

          


    
    elif (dataset == 'naive') & recompute_data_selection:
        dat_type = 'naive-total'
        expList = get_data_bunch(dat_type)
        expList = expList.drop_duplicates(subset=['subject', 'expDate'])

    elif not recompute_data_selection:
        expList = pd.read_csv(savepath)
    
    if (dataset == 'naive') or (not recompute_data_selection):

        recordings = [load_data(subject = rec.subject,
                                expDate = rec.expDate,
                                expNum = rec.expNum,
                                cam_hierarchy= ['frontCam','eyeCam','sideCam'],
                                **current_params) for _,rec in expList.iterrows()]    
        
        recordings = pd.concat(recordings)


    
    recordings[['subject','expDate','expNum']].to_csv(savepath)

    return recordings
