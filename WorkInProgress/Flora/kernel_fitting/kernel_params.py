
def get_params(call_data=True,call_fit=True,call_eval=True): 
    dat_params,fit_params,eval_params = None, None, None 
    if call_data:
        dat_params = {
            't_support_stim':[-0.05,0.6],   
            't_support_movement':[-0.1,0.1],
            'rt_params':{'rt_min': .1, 'rt_max': .6},
            'event_types': ['aud','vis','baseline','move'], # 
            'contrasts': [0.1,0.2,0.4],
            'spls': [0.1],
            'vis_azimuths': [-60,60], 
            'aud_azimuths': [-60,0,60],
            # 'vis_azimuths': [-90,-60,-30,0,30,60,90],
            # 'aud_azimuths': [-90,-60,-30,0,30,60,90],
            'digitise_cam': False,
            'zscore_cam': 'mad'
            }

    if call_fit:
        fit_params = {
            'method':'Ridge',
            'ridge_alpha':1,
            'tune_hyper_parameter':False,
            'rank':10,
            'rr_regulariser':0
        }

    if call_eval:
        eval_params = {
            'kernel_selection':'stimgroups',
            'sig_metric': ['explained-variance']
        }

    return dat_params,fit_params,eval_params