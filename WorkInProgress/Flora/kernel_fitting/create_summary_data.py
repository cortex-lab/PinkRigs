# this is the anatomy figure
# general loading functions
# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from Admin.csv_queryExp import queryCSV


subject_set = 'AV025'
my_expDef = 'postactive'
probe = 'probe1'
recordings = queryCSV(subject = subject_set,expDate='postImplant',expDef=my_expDef)


recordings = recordings[(recordings.extractEvents=='1') & (recordings.extractSpikes=='1,1')]
recordings  = recordings[['subject','expDate','expNum']]
recordings['probe'] = probe

dataset = subject_set + my_expDef + probe

fit_tag = 'stimChoice'

if 'AV025' in subject_set:
    recordings = recordings.iloc[1:-1]

if 'AV034' in subject_set: 
    recordings = recordings.iloc[:-2]
# %%

#  
rerun_sig_test= False 
recompute_csv = True 
recompute_pos_model = False 

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
from Analysis.neural.utils.data_manager import load_cluster_info
from Analysis.neural.src.movements import movement_correlation


# load the ap pref azimuth model 
modelpath = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\anatomy')
modelpath = modelpath / 'aphemi_preferred_azimuth.pickle'
if modelpath.is_file():
    openpickle = open(modelpath,'rb')
    pos_azimuth_fun = pickle.load(openpickle)
else: 
    print('position to azimuth mapping does not exist.')

m = movement_correlation()
# 
interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
csv_path = interim_data_folder / dataset 
csv_path.mkdir(parents=True,exist_ok=True)
csv_path = csv_path / 'summary_data.csv'

tuning_types = ['vis','aud']
cv_names = ['train','test']
if not csv_path.is_file() or recompute_csv:
    all_dfs = []
    for _,session in recordings.iterrows():
    # get generic info on clusters 
        print(*session)

        ################## MAXTEST #################################
        clusInfo = load_cluster_info(**session)
        ################### KERNEL FIT RESULTS #############################
        foldertag = r'kernel_model\%s' % fit_tag
        csvname = '%s_%s_%s_%s.csv' % tuple(session)
        kernel_fit_results = interim_data_folder / dataset  / foldertag / csvname

        kernel_events_to_save = ['aud', 'baseline', 'move_kernel', 'vis','move_kernel_dir']
        for k in kernel_events_to_save:
            tag = 'kernelVE_%s' % k 
            if kernel_fit_results.is_file():
                kernel_fits = pd.read_csv(kernel_fit_results)
                kernel_events_to_save  = np.unique(kernel_fits.event)
                # match neurons 
                curr_set = kernel_fits[(kernel_fits.event==k) & (kernel_fits.cv_number==1)]             

                # concatenate with clusInfo
                unmatched_clus_idx = np.setdiff1d(clusInfo._av_IDs,curr_set.clusID)
                if len(unmatched_clus_idx)==0:
                    clusInfo[tag] = curr_set.VE.values
                else:
                    VEs = curr_set.VE.values
                    newVE = []
                    matched_clusIDs = curr_set.clusID
                    for c in clusInfo._av_IDs:
                        idx = np.where(matched_clusIDs==c)[0]
                        if len(idx)==1:
                            newVE.append(VEs[idx[0]])
                        else:
                            newVE.append(np.nan)  
                    
                    clusInfo[tag] = newVE                     

            else: 
                clusInfo[tag] = np.nan


        all_dfs.append(clusInfo)
    
# temproary hack 
#all_dfs = [d.drop(columns=['sc_azimuth', 'sc_elevation', 'sc_surface']) if 'sc_azimuth' in d.columns else d for d in all_dfs]
clusInfo = pd.concat(all_dfs,axis=0)    
# %%
if csv_path.is_file():
    # save previous
    old = pd.read_csv(csv_path)
    time_created = datetime.datetime.fromtimestamp(
        csv_path.stat().st_ctime
        ).strftime("%Y-%m-%d-%H%M")
    old_save_path = csv_path.parent / ('summary_data%s.csv' % time_created)
    old.to_csv(old_save_path)

clusInfo.to_csv(csv_path)

# %%
