
import pyddm
import numpy as np
import pickle
import matplotlib.pyplot as plt 

from sklearn.model_selection import StratifiedShuffleSplit
from pyddm.functions import solve_partial_conditions

class DriftAdditive(pyddm.Drift):
    name = "additive drift"
    required_parameters = ["aud_coef", "vis_coef","contrast_power"]
    required_conditions = ["audDiff", "visDiff"]
    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        return (self.aud_coef * conditions["audDiff"] + self.vis_coef * visSide * (visContrast**self.contrast_power))
    

class DriftAdditiveSplitOpto(pyddm.Drift):
    name = "DriftAdditiveSplit"
    required_parameters = ["aud_coef_left", "vis_coef_left","aud_coef_right","vis_coef_right","contrast_power","laser_bias"]
    required_conditions = ["audDiff", "visDiff","is_laserTrial"]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        
        myDrift = (self.aud_coef_right * (audSide>0) - self.aud_coef_left * (audSide<0) + 
                    self.vis_coef_right * (visSide>0) * (visContrast**self.contrast_power) - self.vis_coef_left * (visSide<0) * (visContrast**self.contrast_power)) + self.laser_bias * conditions['is_laserTrial'].astype('int')

        return myDrift
 

class DriftAdditiveOpto(pyddm.Drift):
    name = "additive drift"
    required_parameters = ["aud_coef", "vis_coef","contrast_power","laser_bias"]
    required_conditions = ["audDiff", "visDiff","is_laserTrial"]
    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        return (self.aud_coef * conditions["audDiff"] + self.vis_coef * visSide * (visContrast**self.contrast_power)) + self.laser_bias * conditions['is_laserTrial'].astype('int')



class DriftAdditiveSplit(pyddm.Drift):
    name = "DriftAdditiveSplit"    
    required_parameters = ["aL", "vL","aR","vR","gamma","b"]
    required_conditions = ["audDiff", "visDiff"]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        
        myDrift = (self.aR * (audSide>0) - self.aL * (audSide<0) + 
                   self.vR * (visSide>0) * (visContrast**self.gamma) - self.vL * (visSide<0) * (visContrast**self.gamma) +
                   self.b)

        return myDrift
    


    

def get_param(which='bound',type='fittable',value=[None],modelType='DriftAdditiveSplit'): 
    """
    helper function to add parameters to a dictionary 
    Parameters: 
    -----------
    which: str
        name of the parameter of interest. 
        Options: bound,drfit,noise, overlay,IC, dt,dx, T_dur, choice,names

    type: str 
        which paramtype to get 
        Options: fittable, constant
    
    values: list
        if we want to overwrite the parameter/bounds. List because overlay needs drift/overlay need more than one values 
    """

    if len(value)==1:
        value = [value[0],None,None,None,None,None]
    elif len(value) == 3:
        value = [value[0],value[1],value[2],None,None,None]

    constant_value_options = {
        'bound': pyddm.BoundConstant(B=1), 
        'noise': pyddm.NoiseConstant(noise=1),
        'overlay': pyddm.OverlayChain(overlays=[pyddm.OverlayNonDecision(nondectime=2),pyddm.OverlayExponentialMixture(pmixturecoef=2,rate=1)]), 
        'IC': pyddm.ICPoint(x0=0),
        'dt':.001,
        'dx': .001,
        'T_dur' : 2, 
        'choice_names':('Right','Left')
    }

    set_value_options = {
        'bound': pyddm.BoundConstant(B=value[0]), 
        'noise': pyddm.NoiseConstant(noise=value[0]),
        'overlay': pyddm.OverlayChain(overlays=[pyddm.OverlayNonDecision(nondectime=value[0]),pyddm.OverlayExponentialMixture(pmixturecoef=value[1],rate=value[2])]), 
        'IC': pyddm.ICPoint(x0=value[0]),
        'dt':value[0],
        'dx': value[0],
        'T_dur' : value[0], 
        'choice_names':('Right','Left')    
    }

    fittable_value_options  = {
        'noise':pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=3)), 
        'bound':pyddm.BoundConstant(B=pyddm.Fittable(minval=.1, maxval=2)), 
        'overlay':pyddm.OverlayChain(overlays=[
                pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=.01, maxval=.4)),
                pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=.01, maxval=.3),
                rate=pyddm.Fittable(minval=.1, maxval=2))]),
        'IC':pyddm.ICPoint(x0=pyddm.Fittable(minval=-.9, maxval=.9))
    }


 
    if 'DriftAdditiveSplit' in modelType:
        constant_value_options.update({'drift':DriftAdditiveSplit(aud_coef_right=1,
                                aud_coef_left=1,
                                vis_coef_right=1,
                                vis_coef_left=1,
                                contrast_power =1,bias=0)})

        set_value_options.update({'drift':DriftAdditiveSplit(aud_coef_right=value[0],
                                aud_coef_left=value[1],
                                vis_coef_right=value[2],
                                vis_coef_left=value[3],
                                contrast_power = value[4],bias = value[5])})

        fittable_value_options.update({'drift':DriftAdditiveSplit(aud_coef_right=pyddm.Fittable(minval=.01, maxval=8),
                                aud_coef_left=pyddm.Fittable(minval=.01, maxval=8),
                                vis_coef_right=pyddm.Fittable(minval=.01, maxval=8),
                                vis_coef_left=pyddm.Fittable(minval=.01, maxval=8),
                                contrast_power = pyddm.Fittable(minval=.02, maxval=4),
                                bias = pyddm.Fittable(minval=.02, maxval=4))})
        
    elif 'DriftAdditiveJoint' in modelType: 
        pass


    # first contstruct the drift module



    if 'fittable' in type:
        d = {which:fittable_value_options[which]}
    else:
        if value[0] is None:
            d = {which:constant_value_options[which]}
        else: 
            d = {which:set_value_options[which]}    

    return d

# def get_default_parameters(modelType='DriftAdditiveSplit'): 
#     """
#     default parameter set to initialise the fitting

#     """
#     driftclass = getattr(,modelType)

#     params = np.array(['aL','aR','vL','vR','gamma','b', 'noise', 'bound', 'IC'])

#     params = {
#         'bound': pyddm.BoundConstant(B=1), 
#         'noise': pyddm.NoiseConstant(noise=1),
#         'overlay': pyddm.OverlayChain(overlays=[pyddm.OverlayNonDecision(nondectime=2),pyddm.OverlayExponentialMixture(pmixturecoef=2,rate=1)]), 
#         'IC': pyddm.ICPoint(x0=0),
#         'dt':.001,
#         'dx': .001,
#         'T_dur' : 2, 
#         'choice_names':('Right','Left')
#     }


#     pvals = [[None] for i in range(len(ptypes))]
#     new_params = {n:r for p,t,v in zip(pl,ptypes,pvals) for n,r in (get_param(which = p,  type= t,value= v,modelType=modelType)).items()}    
#     return new_params

def get_params_fixation(model,freeP = [0,0,0,0,0,0,0,1,0]):
    """
    function to prepare parameter set by fixing certain parameters while allowing others to readjust 
    freeP: list of bool 
        which parameter is allowed to adjust. Order: drift, noise, bound, IC, overlay

    """

    params = np.array(['aL','aR','vL','vR','gamma','b', 'noise', 'bound', 'IC'])

    ptypes = ['fittable' if cfp==1 else 'constant' for cfp in freeP]

    pvals = [[model.parameters()[mypar][k].real for k in list(model.parameters()[mypar].keys())] if (ptypes[idx]!='fittable') else [None] for idx,mypar in enumerate(pl)]

    # if the bound is fittable but the x0 is fixed we need to ensure that the min(bound)>x0

    new_params = {n:r for p,t,v in zip(pl,ptypes,pvals) for n,r in (get_param(which = p,  type= t,value= v,modelType='DriftAdditiveSplit')).items()}   

    if (freeP[2]==1) & (freeP[3]==0): 
        #fixing starting point, but fitting bound -- need to ensure correct minval.'
        new_params['bound'] = pyddm.BoundConstant(B=pyddm.Fittable(minval=(np.abs(pvals[3][0])*1.3), maxval=3))
    elif (freeP[2]==1) & (freeP[3]==1): 
        #fixing starting point, but fitting bound -- need to ensure correct minval.'
        new_params['bound'] = pyddm.BoundConstant(B=pyddm.Fittable(minval=1.05, maxval=3))


    pl = ['dx','dt','T_dur','choice_names' ]
    ptypes = ['constant','constant','constant','constant','constant']
    pvals = [[None] for i in range(len(ptypes))]
    fixed_params = {n:r for p,t,v in zip(pl,ptypes,pvals) for n,r in (get_param(which = p,  type= t,value= v,modelType='DriftAdditiveSplit')).items()}     

    new_params.update(fixed_params)

    return new_params




def plot_diagnostics(model=None,sample = None, conditions=None,data_dt =.025,method=None,myloc=0,ax = None,dkwargs = None,mkwargs =None):
    """
    visually assess the diagnostics of the model fit

    """
    if not ax:
        _,ax = plt.subplots(1,1)

    if not dkwargs:
        dkwargs = {
            'alpha' : .5, 
            'color' : 'k'
        }

    if not mkwargs: 
        mkwargs = {
            'lw': 2, 
            'color': 'k'
        }

    if model:
        T_dur = model.T_dur
        if model.dt > data_dt:
            data_dt = model.dt
    elif sample:
        T_dur = max(sample)
    else:
        raise ValueError("Must specify non-empty model or sample in arguments")

    # If a sample is given, plot it behind the model.
    if sample:
        s = sample.subset(**conditions)
        t_domain_data = np.linspace(0, T_dur, int(T_dur/data_dt+1))
        data_hist_top = np.histogram(s.choice_upper, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        data_hist_bot = np.histogram(s.choice_lower, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        total_samples = len(s)
        ax.fill_between(np.asarray(data_hist_top)/total_samples/data_dt+myloc,t_domain_data, label="Data",**dkwargs)
        ax.fill_between(-np.asarray(data_hist_bot)/total_samples/data_dt+myloc,t_domain_data, label="Data", **dkwargs)
        toplabel,bottomlabel = sample.choice_names
    if model:
        s = solve_partial_conditions(model, sample, conditions=conditions, method=method)
        ax.plot(s.pdf("_top")+myloc,model.t_domain(),**mkwargs)
        ax.plot(-s.pdf("_bottom")+myloc,model.t_domain(), **mkwargs)
        toplabel,bottomlabel = model.choice_names

