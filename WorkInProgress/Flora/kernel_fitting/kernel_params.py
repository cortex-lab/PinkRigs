def get_params(call_data=True,call_fit=True,call_eval=True,dataset_type = 'naive'): 
    dat_params,fit_params,eval_params = None, None, None 
    if call_data:
        if  dataset_type=='naive':
            dat_params = {
                't_support_stim':[-0.05,0.6],   
                'rt_params':{'rt_min': None, 'rt_max': None},
                'event_types': ['aud','vis','baseline','motionEnergy'],
                # 'vis_azimuths': [-90,-60,-30,0,30,60,90],
                # 'aud_azimuths': [-90,-60,-30,0,30,60,90],
                'vis_azimuths': [-60,0,60], 
                'aud_azimuths': [-60,0,60],
                'contrasts': 'all',
                'spls':'all',
                'zscore_cam': 'mad',
                'turn_stim_off' : None
            }


        elif dataset_type=='active': 
            dat_params = {
                't_support_stim':[-0.15,.6],   
                't_support_movement':[-0.15,0.1],
                'rt_params':{'rt_min': 0.06, 'rt_max': .6},
                'event_types': ['aud','vis','move'], # 
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
                'vis_dir_kernel': False
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
        if  dataset_type=='naive':

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