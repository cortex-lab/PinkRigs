def get_params(call_data=True,call_fit=True,call_eval=True,dataset_type = 'naive'): 

    """
    Generate parameter dictionaries for data loading, model fitting, and evaluation.

    Parameters:
    call_data : bool, optional
        Whether to generate parameters for data loading (default is True).
    call_fit : bool, optional
        Whether to generate parameters for model fitting (default is True).
    call_eval : bool, optional
        Whether to generate parameters for evaluation (default is True).
        
    dataset_type : str, optional
        Type of dataset to generate parameters for. Options include 'naive', 'passive', and 'active' (default is 'naive').

    Returns:
    tuple:
        dat_params : dict or None
            Parameters for data loading based on the dataset type.
        fit_params : dict or None
            Parameters for model fitting.
        eval_params : dict or Nonea
            Parameters for evaluation.
    """

    dat_params,fit_params,eval_params = None, None, None 
    if call_data:
        if  dataset_type=='naive':
            dat_params = {
                't_support_stim':[-0.05,0.3],   
                'rt_params':{'rt_min': None, 'rt_max': None},
                'event_types': ['aud','vis','baseline','motionEnergy','blank'],
                'vis_azimuths': [-90,-60,-30,0,30,60,90],
                'aud_azimuths': [-90,-60,-30,0,30,60,90],
                #'vis_azimuths': [-60,0,60], 
                #'aud_azimuths': [-60,0,60],
                'contrasts': 'all',
                'contra_vis_only':False,
                'spls':'all',
                'zscore_cam': 'mad',
                'turn_stim_off' : None,
                'aud_dir_kernel': False, 
                'vis_dir_kernel': False,
                'contra_vis_only':False
            }

        elif  dataset_type=='passive':
            dat_params = {
                't_support_stim':[-0.05,0.6],   
                'rt_params':{'rt_min': None, 'rt_max': None},
                'event_types': ['aud','vis','baseline','motionEnergy','blank'],
                # 'vis_azimuths': [-90,-60,-30,0,30,60,90],
                # 'aud_azimuths': [-90,-60,-30,0,30,60,90],
                'vis_azimuths': [-60,0,60], 
                'aud_azimuths': [-60,0,60],
                'contrasts': 'all',
                'contra_vis_only':False,
                'spls':'all',
                'zscore_cam': 'mad',
                'turn_stim_off' : None,
                'aud_dir_kernel': False, 
                'vis_dir_kernel': False,
                'contra_vis_only':False
            }


        elif dataset_type=='active': 
            dat_params = {
                't_support_stim':[-0.05,0.5],   
                't_support_movement':[-0.15,0.1],
                'rt_params':{'rt_min': 0.01, 'rt_max': 1.5},
                'event_types': ['aud','vis','move','baseline','motionEnergy'], # 
                'contrasts': 'all', # can also be a list of specified values
                'spls': 'all',
                # 'vis_azimuths': 'dir', 
                # 'aud_azimuths': 'dir',            
                'vis_azimuths': [-60,60], 
                'aud_azimuths': [-60,0,60],
                # 'vis_azimuths': [-90,-60,-30,0,30,60,90],
                # 'aud_azimuths': [-90,-60,-30,0,30,60,90],
                'digitise_cam': False,
                'zscore_cam': 'mad',
                'turn_stim_off' : 'moveEnd',
                'aud_dir_kernel': True, 
                'vis_dir_kernel': True,
                'contra_vis_only':True
                }

    if call_fit:
        fit_params = {
            #'method':'ReduceThenElasticNetCV',
            'method':'Ridge',            
            'ridge_alpha':1,
            'tune_hyper_parameter':False,
            'rank':10,
            'rr_regulariser':0, 
            'l1_ratio': 0
        }

    if call_eval:
        if  (dataset_type=='naive') or (dataset_type=='passive'):

            eval_params = {
                'kernel_selection':'stimgroups',
                'sig_metric': ['explained-variance']
            }
        
        elif dataset_type=='active':
            eval_params = {
                'kernel_selection':'dirgroups',
                'sig_metric': ['explained-variance']
            }

    return dat_params,fit_params,eval_params