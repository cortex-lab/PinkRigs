# functions that load and preselect data for analyses


import sys
from pathlib import Path
import pandas as pd 
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.ev_dat import filter_active_trials


def call_neural_dat(dataset='naive',spikeToInclde=True, camToInclude = True, recompute_data_selection=False,**kwargs):
    """function to load data that requires Video data."""

    data_name_dict = { 'events': {'_av_trials': 'table'}}
    cam_dict = {'frontCam':{'camera':['times','ROIMotionEnergy']},
                            'eyeCam':{'camera':['times','ROIMotionEnergy']},
                            'sideCam':{'camera':['times','ROIMotionEnergy']}}
    ephys_dict = {'spikes':'all','clusters':'all'}
    ephys_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 


    savepath = r'D:\VideoAnalysis\%s_dataset.csv' % dataset

    
    if 'naive' in dataset:
        subject_list = ['FT008','FT009','FT010','FT011','FT019','FT022','FT025','FT027','FT038','FT039']
    elif 'all' in dataset: 
        subject_list = 'all'
    else:
        subject_list = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034']

    
    if 'naive' in dataset and camToInclude:
        cam_hierarchy = ['frontCam','eyeCam','sideCam']
    elif 'naive' not in dataset and camToInclude:
        cam_hierarchy =  ['sideCam','eyeCam','frontCam']
    else: 
        cam_hierarchy = None

    
    if camToInclude:
        data_name_dict.update(cam_dict)

    if spikeToInclde:
        data_name_dict.update(ephys_dict)

    
    query_params = {
        'subject': subject_list,
        'data_name_dict':data_name_dict,
        'checkEvents':'1', 
        'cam_hierarchy': cam_hierarchy
    }

    # in case I want ot call any further parameter in addiiton
    query_params.update(kwargs)


    if (dataset == 'naive') & recompute_data_selection:   
        recordings = load_data(expDef = ['AVPassive_spatialIntegrationFlora_with_ckeckerboard',
                                        'AVPassive_ckeckerboard_updatechecker',
                                        'AVPassive_ckeckerboard_extended',
                                        'AVPassive_checkerboard_extended'],
                                            **query_params)
        
    elif (dataset=='allPassive') & recompute_data_selection:
        recordings = load_data(expDef = ['AVPassive','postactive'],
                                            **query_params)
        
    elif (dataset == 'postactive') & recompute_data_selection:
        recordings = load_data(expDef = 'postactive',expDate = 'postImplant', **query_params)

    elif (dataset == 'active') & recompute_data_selection:
        recordings = load_data(expDef = 'multiSpaceWorld',expDate = 'postImplant', **query_params)  

        exclude_premature_wheel = False
        rt_params = {'rt_min':0.03,'rt_max':1.5}

        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]  


    elif not recompute_data_selection:
        expList = pd.read_csv(savepath)           

        recordings = [load_data(subject = rec.subject,
                                expDate = rec.expDate,
                                expNum = rec.expNum,
                                cam_hierarchy = cam_hierarchy,
                                **query_params) for _,rec in expList.iterrows()]    
        
        recordings = pd.concat(recordings)

    if spikeToInclde:
        # basically double check spiking because the extractSpikes==1 is not enough. Some alignments are weird
        # see e.g. AV025/2022-11-07/4
        is_good_spike = np.array([len(rec.probe.spikes.clusters)>0 for _,rec in recordings.iterrows()])
        recordings = recordings.iloc[is_good_spike] 


    recordings[['subject','expDate','expNum']].to_csv(savepath)

    return recordings
