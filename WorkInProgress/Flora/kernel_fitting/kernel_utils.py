
import sys,shutil
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.io import save_dict_to_json
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.neural.utils.data_manager import load_cluster_info
from Analysis.neural.src.kernel_model import kernel_model
from kernel_params import get_params

def fit_and_save(recordings,recompute=True,savepath=None,dataset_name = 'whoKnows',fit_tag = 'additive-fit',**param_tags): 
    """
    queryCSV based fitting procedure. 

    Parameters:

    """

    if savepath is None: 
        interim_data_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\Audiovisual')
        save_path = interim_data_folder / dataset_name / 'kernel_model' / fit_tag
        save_path.mkdir(parents=True,exist_ok=True)

    failed_recs = []
    for _,rec_info in recordings.iterrows():

        try: 
            print('Now attempting to fit %s %s, expNum = %s, %s' % tuple(rec_info))

            nametag = '%s_%s_%s_%s' % tuple(rec_info)

            # create a folder 
            curr_save_path  = save_path / nametag

            if not curr_save_path.is_dir() or recompute:
                
                # remove if the folder already exists
                if curr_save_path.is_dir():
                    shutil.rmtree(curr_save_path)
                
                curr_save_path.mkdir(parents=True,exist_ok=True)

                # since kernels is a class, I think it is safer to recall it after each fit... I think that is why -1000 kept accumulating in dat_params, for example
                kernels = kernel_model(t_bin=0.005,smoothing=0.025)
                dat_params,fit_params,eval_params = get_params(**param_tags)
                kernels.load_and_format_data(**dat_params,**rec_info)
                kernels.fit(**fit_params)
                variance_explained = kernels.evaluate(**eval_params)


                # save all the results
                variance_explained.to_csv((curr_save_path / ('variance_explained_batchKernel.csv')))
                my_kernels = kernels.calculate_kernels()

                for k in my_kernels.keys():
                    np.save(
                        (curr_save_path / ('%s.npy' % k)), 
                        my_kernels[k]
                    )
                
                np.save(
                    (curr_save_path / 'clusIDs.npy'), 
                    kernels.clusIDs
                )

        except:
            print('Failed to fit %s %s, expNum = %s, %s' % tuple(rec_info))
            failed_recs.append(rec_info)


    failed_recs = pd.DataFrame(failed_recs,columns = ['subject','expDate','expNum','probe'])

    failed_recs.to_csv((save_path / 'failed_to_fit.csv'))    
    # save the parameters of fitting
    save_dict_to_json(dat_params,save_path / 'dat_params.json')
    save_dict_to_json(fit_params,save_path / 'fit_params.json')
    save_dict_to_json(eval_params,save_path / 'eval_params.json')





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
    dat_keys = get_data_bunch(dataset_name)

    all_dfs = []
    for _,rec_info in dat_keys.iterrows():
        # get generic info on clusters 
        print(*rec_info)
        clusInfo = load_cluster_info(**rec_info,unwrap_independent_probes=False)
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