import pyddm
import model_components as model_components
import numpy as np

def get_parameters(freePs=None): 
    """
    build model from parameter units
    """

    unit_classes =  {
            'drift': getattr(model_components,'DriftAdditiveOpto'),
            'noise': model_components.get_default_noise(),
            'bound': getattr(model_components,'BoundOpto'),
            'nondectime': getattr(model_components,'OverlayNonDecisionOpto'),
            'mixture':getattr(model_components,'OverlayExponentialMixtureOpto'),
            'IC': getattr(model_components,'ICPointOpto')
        }
    
    keys = list(unit_classes.keys())
    # overwrite default free Parameters 
    if freePs: 
        for k in list(freePs.keys()):
            unit_classes[k].freeP = freePs[k]     

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

