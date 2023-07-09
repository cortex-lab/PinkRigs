# this is the anatomy figure
# general loading functions
# %%
import sys,datetime,pickle
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
subject_set = ['AV025','AV030','AV034']
my_expDef = 'multiSpaceWorld_checker_training'
subject_string = ''.join(subject_set)
dataset = subject_string + my_expDef

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')

csv_path = interim_data_folder / dataset 
fit_tag = 'stimChoice'
foldertag = r'kernel_model\%s' % fit_tag
kernel_fit_results = interim_data_folder / dataset  / foldertag 


subfolders = [item for item in kernel_fit_results.iterdir() if item.is_dir()]

recordings = [tuple(s.name.split('_')) for s in subfolders]
recordings = pd.DataFrame(recordings,columns=['subject','expDate','expNum','probe'])


# %%
from Analysis.neural.utils.data_manager import load_cluster_info

#%%

recompute_csv = True

csv_path = csv_path / 'summary_data.csv'

tuning_types = ['vis','aud']
cv_names = ['train','test']
if not csv_path.is_file() or recompute_csv:
    all_dfs = []
    for (_,session),results_folder in zip(recordings.iterrows(),subfolders):
    # get generic info on clusters 
        print(*session)

        ################## MAXTEST #################################
        clusInfo = load_cluster_info(**session)
        ################### KERNEL FIT RESULTS #############################

        ve_results = results_folder / 'variance_explained_batchKernel.csv'


        kernel_events_to_save = ['aud_kernel_spl_0.10','aud_kernel_spl_0.10_dir', 'baseline', 'move_kernel', 'vis','move_kernel_dir']
        for k in kernel_events_to_save:
            tag = 'kernelVE_%s' % k 
            if ve_results.is_file():
                kernel_fits = pd.read_csv(ve_results)
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


        # add the average of the kernels
        clusIDs = np.load(list(results_folder.glob('clusIDs.npy'))[0]) 
        kernel_files = list(results_folder.glob('*kernel*.npy')) 
        n_bins = 50
        for k in kernel_files:
            my_k = np.load(k)
            if 'move' in k.stem:
                my_k = my_k[:,::-1] # reverse along time
            
            sumkernel = (my_k[:,:n_bins]).sum(axis=1)

            # I really should make this a function...
            sumkernel_ = []
            for c in clusInfo._av_IDs:
                idx = np.where(clusIDs==c)[0]
                if len(idx)==1:
                    sumkernel_.append(sumkernel[idx[0]])
                else:
                    sumkernel_.append(np.nan)

            tag = 'kernelSum_' + k.stem

            clusInfo[tag] = sumkernel_





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
