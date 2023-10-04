import pyddm
import model_components as model_components
import numpy as np

def get_parameters(driftType='DriftAdditiveSplit',nondectimeType=None,model=None,freePs=None): 
    """
    build model from parameter units

    """

    if nondectimeType is not None: 
        nondectime = getattr(model_components,nondectimeType)
    else:
        nondectime = model_components.get_default_nondecision()

    unit_classes =  {
            'drift': getattr(model_components,driftType),
            'noise': model_components.get_default_noise(),
            'bound': model_components.get_default_bound(),
            'nondectime': nondectime,
            'mixture':model_components.get_default_mixture(),
            'IC': model_components.get_default_IC()
        }
    

    keys = list(unit_classes.keys())


    if model: 
        paramvals  = {mypar:[model.parameters()[mypar][k].real for k in list(model.parameters()[mypar].keys())] for idx,mypar in enumerate(['drift','noise','bound','IC'])}
        if nondectimeType is None: 
            paramvals.update({'nondectime':[model.parameters()['overlay']['nondectime'].real],
                            'mixture':[model.parameters()['overlay']['pmixturecoef'].real,model.parameters()['overlay']['rate'].real]})
        else: 
            paramvals.update({'nondectime':[model.parameters()['overlay']['nondectimeA'].real,model.parameters()['overlay']['nondectimeV'].real],
                            'mixture':[model.parameters()['overlay']['pmixturecoef'].real,model.parameters()['overlay']['rate'].real]})
        # update the fixed parameters
        for k in keys: 
            unit_classes[k].fixedC = paramvals[k]

        if freePs: 
            for k in list(freePs.keys()):
                unit_classes[k].freeP = freePs[k]     

    # initialise the classes params
    
    class_params = {k:{p:pyddm.Fittable(minval=mi, maxval=ma) if to_fit else c 
                       for p,mi,ma,c,to_fit in 
                       zip(unit_classes[k].required_parameters,
                           unit_classes[k].fittable_minvals,
                           unit_classes[k].fittable_maxvals,
                           unit_classes[k].fixedC,
                           unit_classes[k].freeP)}                        
                       for k in keys}

    params = {
        'drift': unit_classes['drift'](**class_params['drift']),
        'noise': unit_classes['noise'](**class_params['noise']), 
        'bound': unit_classes['bound'](**class_params['bound']), 
        'overlay':pyddm.OverlayChain(overlays=[unit_classes['nondectime'](**class_params['nondectime']),unit_classes['mixture'](**class_params['mixture'])]),
        'IC':unit_classes['IC'](**class_params['IC']),
        'dt':.001,
        'dx': .001,
        'T_dur' : 2, 
        'choice_names':('Right','Left')
    }


    return params

