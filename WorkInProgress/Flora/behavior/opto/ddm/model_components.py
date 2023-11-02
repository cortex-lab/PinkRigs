"""
script that includes model components for fitting.
# each class is a container for parameter components. Each class must contain the following objects: 

required_parameters: list of str (n)
required_conditions: str (m)
fittable_minvals: list of floats (n)
    minval for fitting parameter
fittable_maxvals: list of floats (n)
    maxval for fitting parameter
fixedC: list of floats (n)
    constant value the param should take 
freeP: list of floats (n)
    whether parameter should take the constant or the fittable 

"""

import numpy as np 
import pyddm

# each class is a set of parameter components. Each class must contain the following objects: 
# class DriftAdditiveOpto(pyddm.Drift):
#     name = "DriftAdditiveSplit"    
#     required_parameters = [
#         "aL", "vL","aR","vR","gamma","b", 
#         "d_aL", "d_vL","d_aR","d_vR" 
#         ]    
#     required_conditions = ["audDiff", "visDiff",'is_laserTrial']

#     fittable_minvals = [
#         .01,.01,.01,.01,0.5,-8,
#         .01,.01,.01,.01,-8]
#     fittable_maxvals = [
#         8,8,8,8,1.5,8, 
#         8,8,8,8,8]
#     fixedC = [1,1,1,1,1,0,
#               0,0,0,0]
#     freeP = [1,1,1,1,1,0,
#              0,0,0,0]

#     def get_drift(self, conditions, **kwargs):
#         visContrast = np.abs(conditions["visDiff"])
#         visSide = np.sign(conditions["visDiff"])
#         audSide = np.sign(conditions["audDiff"])
#         isOpto = conditions['is_laserTrial']
        
#         myDrift = ((self.aR + self.d_aR *isOpto) * (audSide>0) - (self.aL + self.d_aL * isOpto) * (audSide<0) + 
#                    (self.vR + self.d_vR) * (visSide>0) * (visContrast**self.gamma) - (self.vL+self.d_vL*isOpto) * (visSide<0) * (visContrast**self.gamma) +
#                    (self.b *isOpto))

#         return myDrift
    
class DriftAdditiveOpto(pyddm.Drift):
    name = "DriftAdditiveSplit"    
    required_parameters = [
        "a", "v","aS","vS","gamma", "b", #parameters applied to all (6)
        "d_aR","d_aL", "d_vR","d_vL","d_b"  #opto dependent parameters (5)
        ]    
    required_conditions = ["audDiff", "visDiff",'is_laserTrial']

    fittable_minvals = [
        .01,.01,-4,-4,0.5,-4,
        -3,-3,-3,-3,-6]
    fittable_maxvals = [
        6,6,4,4,1.5,4, 
        6,6,6,6,6]
    fixedC = [1,1,1,1,1,0,
              0,0,0,0,0]
    
    freeP = [1,1,1,1,1,0,
             0,0,0,0]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        isOpto = conditions['is_laserTrial']
        
        # variables
        a_R = (audSide>0)
        a_L = (audSide<0)
        v_R = (visSide>0) * (visContrast**self.gamma)
        v_L = (visSide<0) * (visContrast**self.gamma)
        audComponent = (self.a + self.d_aL * isOpto) * a_R - (self.a + self.aS + self.d_aR * isOpto) * a_L
        visComponent = (self.v + self.d_vL * isOpto) * v_R - (self.v + self.vS + self.d_vR * isOpto) * v_L
        biasComponent = self.b + self.d_b* isOpto

        myDrift = audComponent + visComponent + biasComponent

        return myDrift

class BoundOpto(pyddm.Bound): 
    name = 'constant bound that can expland by opto'
    required_parameters = ["B", "d_B"]
    required_conditions = ["is_laserTrial"] 
    fittable_minvals = [.9,0]
    fittable_maxvals = [1.1,4]
    fixedC = [1,0]

    def get_bound(self,conditions,*args,**kwargs):
        isOpto = conditions['is_laserTrial']

        return  self.B + (self.d_B * isOpto)


class OverlayNonDecisionOpto(pyddm.OverlayNonDecision):
    name = "Separate non-decision time for aud and vis components"
    required_parameters = ["nondectime", "d_nondectimeOpto"]
    required_conditions = ["is_laserTrial"] 
    fittable_minvals = [.01,-.4]
    fittable_maxvals = [.4,.4]
    fixedC = [.3,0]
    freeP = [1,0]    

    def get_nondecision_time(self, conditions):
        isOpto = conditions['is_laserTrial']
        return self.nondectime + self.d_nondectimeOpto * isOpto

class ICPointOpto(pyddm.ICPoint):
    name = "A starting point with a left or right bias."
    required_parameters = ["x0", "d_x0"]
    required_conditions = ["is_laserTrial"]
    fittable_minvals = [-.9,-.9]
    fittable_maxvals = [.9,.9]
    fixedC = [0,0]
    freeP = [1,0]

    def get_starting_point(self, conditions):
        isOpto = conditions['is_laserTrial']

        start = self.x0+(self.d_x0*isOpto)
        # we fix bound and if .95 exceeded we fix the start        
        if start>0.95:
            start=.95
        elif start<-.95:
            start =-.95
        return start


class OverlayExponentialMixtureOpto(pyddm.Overlay):
    """An exponential mixture distribution where the mixture coef depends on opto

    """
    name = "Exponential distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef", "rate","d_pmixturecoef"]
    required_conditions = ["is_laserTrial"]
    fittable_minvals = [0.01,.3,-.5]
    fittable_maxvals = [.5,2,.5]
    fixedC = [.2,1,0]
    freeP = [1,1,0]

    def apply(self, solution):

        assert self.pmixturecoef >= 0 and self.pmixturecoef <= 1
        choice_upper = solution.choice_upper
        choice_lower = solution.choice_lower
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        isOpto = cond['is_laserTrial']

        norm = np.sum(choice_upper)+np.sum(choice_lower)
        lapses = lambda t : 2*self.rate*np.exp(-1*self.rate*t)
        X = m.dt * np.arange(0, len(choice_upper))
        Y = lapses(X)
        Y /= np.sum(Y)
        total_mixture = self.pmixturecoef + self.d_pmixturecoef * isOpto

        # basically we allow mixture to really saturate
        if total_mixture<=0:
            total_mixture=.0001

        choice_upper = choice_upper*(1-total_mixture) + .5*total_mixture*Y*norm # Assume numpy ndarrays, not lists
        choice_lower = choice_lower*(1-total_mixture) + .5*total_mixture*Y*norm
        #print(choice_upper)
        #print(choice_lower)
        return pyddm.Solution(choice_upper, choice_lower, m, cond, undec, evolution)
    
def get_default_noise():
    c = pyddm.NoiseConstant
    c.fittable_minvals = [.2]
    c.fittable_maxvals = [3]
    c.fixedC = [1]
    c.freeP = [1]
    return c

def get_default_drift():
    c = pyddm.DriftConstant
    c.fittable_minvals = [.01]
    c.fittable_maxvals = [3]
    c.fixedC = [.5]
    c.freeP = [1]
    return c


def get_default_bound():
    c = pyddm.BoundConstant
    c.fittable_minvals = [1]
    c.fittable_maxvals = [3]
    c.fixedC = [1]
    c.freeP = [0]
    return c

def get_default_IC():
    c = pyddm.ICPoint
    c.fittable_minvals = [-.9]
    c.fittable_maxvals = [.9]
    c.fixedC = [0]
    c.freeP = [0]
    return c


def get_default_nondecision():
    c = pyddm.OverlayNonDecision
    c.fittable_minvals = [.01]
    c.fittable_maxvals = [.4]
    c.fixedC = [1]
    c.freeP = [1]
    return c

def get_default_mixture():
    c = pyddm.OverlayExponentialMixture
    c.fittable_minvals = [.01,.1]
    c.fittable_maxvals = [.3,2]
    c.fixedC = [.1,1]
    c.freeP = [1,1]
    return c


def get_freeP_sets(which = 'ctrl'):
    """
    hardcoded dictionaries that allow sets of parameters to fix vs fit 

    
    start with 'all'

    Q1 can we get rid of any parameter? 
    'all'
    'l_[param_name]' 

    # lets do it sequentially 
    (some of their combinations)
    'l_sensoryLR'
    [
        "a", "v","aS","vS","gamma","b", #parameters applied to all (6)
        "d_aR","d_aL", "d_vR","d_vL","d_b"  #opto dependent parameters (5)
        ]    

    Q1: do we need all the drift parameters?
    'ctrl'
    'd_b'
    'd_x0'
    'd_x0_d_b'
    'd_mix_d_b'
    'd_S_d_b'
    'd_nondec_d_b'

    """
    if 'all' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }


    elif 'l_aS' in which: 
        freePs = {
            'drift': [1,1,0,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_vS' in which: 
        freePs = {
            'drift': [1,1,1,0,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_gamma' in which: 
        freePs = {
            'drift': [1,1,1,1,0,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_b' in which: 
        freePs = {
            'drift': [1,1,1,1,1,0,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }


    elif 'l_d_aR' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_d_aL' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,0,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_d_vR' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,0,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_d_vL' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,0,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_d_b' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,0], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,1]
        }


    elif 'l_d_nondectime' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,0],
            'mixture':[1,1,1],
                'IC': [1,1]
        }

    elif 'l_d_mixturecoef' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,0],
                'IC': [1,1]
        }


    elif 'l_d_x0' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      1,1,1,1,1], 
            'noise':[1],
            'bound': [0,0], # never changes
            'nondectime':[1,1],
            'mixture':[1,1,1],
                'IC': [1,0]
        }


    elif 'simplest_control' in which: 
        freePs = {
            'drift': [1,1,0,0,0,0,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,0]
        }

    elif 'ctrl' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,0]
        }

    elif 'g_d_b' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,0]
        }

    elif 'g_d_x0' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1]
        }

    elif 'g_both' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,1], 
            'noise':[1],
            'bound': [0,0],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1]
        }


    elif 'g_boundx0' in which: 
        freePs = {
            'drift': [1,1,1,1,1,1,
                      0,0,0,0,0], 
            'noise':[1],
            'bound': [0,1],
            'nondectime':[1,0],
            'mixture':[1,1,0],
                'IC': [1,1]
        }


    elif 'fixed' in which: 
        freePs = {
            'drift': [0,0,0,0,0,0,
                      0,0,0,0,0], 
            'noise':[0],
            'bound': [0,0],
            'nondectime':[0,0],
            'mixture':[0,0,0],
                'IC': [0,0]
        }



    return freePs


