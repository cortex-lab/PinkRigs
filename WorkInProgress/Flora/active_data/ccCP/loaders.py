import sys
from pathlib import Path
import pandas as pd 
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import load_data
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.ev_dat import filter_active_trials,parse_events


def load_for_ccCP(recompute_data_selection=False):
    """function to load the data..."""

    savepath = r'D:\ccCP\dataset.csv'    
    ephys_dict = {'spikes':'all','clusters':'all'}

    current_params = {
    'data_name_dict':{'probe0':ephys_dict,'probe1':ephys_dict,
                        'events': {'_av_trials': 'table'}},
                        'checkSpikes':'1',
                        'unwrap_probes':True,
                        'filter_unique_shank_positions':True,
                        'merge_probes':False,                              
                        'cam_hierarchy': None

    }     

    if recompute_data_selection:
        # get the spikes into the data name dict
   

        recordings = load_data(subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],
                               checkEvents='1',expDate='postImplant',
                               expDef = 'multiSpaceWorld',
                                **current_params) 
        
        exclude_premature_wheel = False
        rt_params = {'rt_min':0.03,'rt_max':1.5}

        # instead of this I would need to do "parse events"

        n_trials  = np.array([filter_active_trials(rec.events._av_trials,
                                                rt_params=rt_params,
                                                exclude_premature_wheel=exclude_premature_wheel).sum()
                                                
                                                for _,rec in recordings.iterrows()])

        recordings = recordings.iloc[n_trials>150]      

   

    else:
        expList = pd.read_csv(savepath)
    

        recordings = [load_data(subject = rec.subject,
                                expDate = rec.expDate,
                                expNum = rec.expNum,
                                **current_params) for _,rec in expList.iterrows()]    
        
        recordings = pd.concat(recordings)


    
    recordings[['subject','expDate','expNum']].to_csv(savepath)

    return recordings
