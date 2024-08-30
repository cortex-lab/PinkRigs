
import sys,shutil,re
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,format_cluster_data
from Analysis.pyutils.io import save_dict_to_json,get_subfolders
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.neural.utils.data_manager import load_cluster_info
from Analysis.neural.src.kernel_model import kernel_model
from kernel_params import get_params

def fit_and_save(recordings,recompute=True,savepath=None,dataset_name = 'active',fit_tag = 'additive-fit',**param_tags): 
    """
    queryCSV based fitting procedure. 

    Parameters:
    recordings : DataFrame
        A DataFrame containing information about the recordings to be processed.
    recompute : bool, optional
        If True, recompute the fits even if they already exist (default is True).
    savepath : str or Path, optional
        Path where the results will be saved. If None, a default path is used (default is None).
    dataset_name : str, optional
        Name of the dataset (default is 'active'). Used to identify the dataset in saving folders.
    fit_tag : str, optional
        Tag used to identify the fitting method (default is 'additive-fit').
    **param_tags : dict
        Additional parameters passed to get_params function from kernel params.
    """

    # If no save path is provided, use a default location.

    if savepath is None: 
        interim_data_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')
        save_path = interim_data_folder / dataset_name / 'kernel_model' / fit_tag
        save_path.mkdir(parents=True,exist_ok=True)

    failed_recs = [] # List to keep track of failed recordings.


    # Iterate over each recording in the provided DataFrame that should be in a queryCSV format
    for _, rec_info in recordings.iterrows():
        # Create a unique identifier (nametag) for each recording based on its information.
        nametag = '%(subject)s_%(expDate)s_%(expNum)s_%(probeID)s' % rec_info
        print('Now attempting to fit %s' % nametag)

        try:  
            curr_save_path = save_path / nametag  # Set the save path for the current recording.

            # Check if the directory already exists or if recomputation is requested.
            if not curr_save_path.is_dir() or recompute:
                
                # If the directory exists and recomputation is required, remove the existing directory.
                if curr_save_path.is_dir():
                    shutil.rmtree(curr_save_path)
                
                curr_save_path.mkdir(parents=True, exist_ok=True)  # Create the directory.
                
                # Initialize the kernel model. Reinitializing helps avoid issues with accumulating values in previous fits.
                kernels = kernel_model(t_bin=0.005, smoothing=0.025)
                dat_params, fit_params, eval_params = get_params(**param_tags)  # Get parameters for the fitting process.

                # Load and format the data based on the recording information.
                if hasattr(rec_info, 'probe'):
                    kernels.load_and_format_data(rec=rec_info, **dat_params)
                else:
                    kernels.load_and_format_data(**dat_params, **rec_info)

                # Fit the model to the data.
                kernels.fit(**fit_params)
                
                # Evaluate the fit and calculate the variance explained.
                variance_explained = kernels.evaluate(**eval_params)

                # Save the variance explained to a CSV file.
                variance_explained.to_csv((curr_save_path / 'variance_explained_batchKernel.csv'))

                # Calculate the kernels and save them as .npy files.
                my_kernels = kernels.calculate_kernels()
                for k in my_kernels.keys():
                    np.save((curr_save_path / ('%s.npy' % k)), my_kernels[k])
                
                # Save the cluster IDs used in the model.
                np.save((curr_save_path / 'clusIDs.npy'), kernels.clusIDs)

        except:
            # If any exception occurs, report the failure and add the recording info to the failed list.
            print('Failed to fit %s' % nametag)
            failed_recs.append(rec_info[['subject', 'expDate', 'expNum','probeID']])

    # Convert the list of failed recordings to a DataFrame.
    failed_recs = pd.DataFrame(failed_recs, columns=['subject', 'expDate', 'expNum','probeID'])

    # Save the failed recordings to a CSV file.
    failed_recs.to_csv((save_path / 'failed_to_fit.csv'))    
    
    # Save the parameters used for fitting as JSON files.
    save_dict_to_json(dat_params, save_path / 'dat_params.json')
    save_dict_to_json(fit_params, save_path / 'fit_params.json')
    save_dict_to_json(eval_params, save_path / 'eval_params.json')





def load_VE_per_cluster(dataset_name,fit_tag,unite_aud=True,interim_data_folder=Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')):
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
    unite_aud: bool 
        whether to unite all aud kernels indepedent of SPL
    
    Returns: pd.df
        clusInfo 

    
    """

    save_path = interim_data_folder / dataset_name / 'kernel_model' / fit_tag
    sess_folders = get_subfolders(save_path)
    dat_keys = [re.split('_', sess.stem) for sess in sess_folders]
    dat_keys = pd.DataFrame(pd.DataFrame(dat_keys,columns = ['subject','expDate','expNum','probe']))

    all_dfs = []
    for _,rec_info in dat_keys.iterrows():
        # get generic info on clusters 
        print(*rec_info)

        data_dict = {
            rec_info.probe:{'clusters':'all','spikes':'clusters'}}
        recording = load_data(data_name_dict=data_dict,**rec_info[['subject','expDate','expNum']])   
        rec = recording.iloc[0]

        clusInfo = format_cluster_data(rec[rec_info.probe].clusters)

        nametag = '%s_%s_%s_%s' % tuple(rec_info)        
        current_folder = save_path / nametag

        kernel_fit_results = current_folder / ('variance_explained_batchKernel.csv')


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
                
                
                if ('aud' in tag) & (unite_aud):
                    if 'dir' in tag:
                        clusInfo['kernelVE_aud_dir'] = clusInfo[tag]
                    else:
                        clusInfo['kernelVE_aud'] = clusInfo[tag]



            # procedure to load in the actual kernels. 
            cIDs = np.load(current_folder / 'clusIDs.npy')       
            kernel_paths = list(current_folder.glob('*kernel*.npy'))
            for k in kernel_paths:
                collected_k = []
                myk = np.load(k)
                for idx,c in enumerate(clusInfo._av_IDs):
                    matrix_idx = np.where(cIDs==c)[0]
                    if matrix_idx.size==1: 
                        matrix_idx = matrix_idx[0]
                        collected_k.append(myk[matrix_idx,:][np.newaxis,:]) 
                    else:      
                        collected_k.append(np.empty((1,myk.shape[1]))*np.nan)  

                collected_k  = np.concatenate(collected_k)
                clusInfo[k.stem] = collected_k.tolist() 

            all_dfs.append(clusInfo)
        
    clusInfo = pd.concat(all_dfs,axis=0)   

    return clusInfo


def write_summary_data(): 
    pass