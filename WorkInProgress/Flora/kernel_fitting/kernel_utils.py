
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.neural.utils.data_manager import load_cluster_info

def load_VE_per_cluster(dataset,fit_tag,interim_data_folder=Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')):
    """"
    function that reads the csv output of the kernel fitting and matches that with the ONE clusters info
    
    Parameters:
    ----------
    dataset: str
        name of dataset in the batch data that we read in
    fit_tag: str
        name of the model during model fitting
    interim_data_folder: pathlib.Path
        parent folder where the fittings are stored
    
    Returns: pd.df
        clusInfo 

    
    """

    save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag
    dat_keys = get_data_bunch(dataset)

    all_dfs = []
    for _,session in dat_keys.iterrows():
        # get generic info on clusters 
        print(*session)
        clusInfo = load_cluster_info(**session,unwrap_independent_probes=False)

        kernel_fit_results = (save_path / ('%s_%s_%.0f_%s.csv' % tuple(session)))

        if kernel_fit_results.is_file():
            kernel_fits = pd.read_csv(kernel_fit_results)  
            kernel_events_to_save  = np.unique(kernel_fits.event)
            for k in kernel_events_to_save:
                tag = 'kernelVE_%s' % k                            
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

            all_dfs.append(clusInfo)
        
    clusInfo = pd.concat(all_dfs,axis=0)   

    return clusInfo