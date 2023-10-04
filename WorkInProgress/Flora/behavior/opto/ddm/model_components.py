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

class DriftAdditiveSplit(pyddm.Drift):
    name = "DriftAdditiveSplit"    
    required_parameters = ["aL", "vL","aR","vR","gamma","b"]
    required_conditions = ["audDiff", "visDiff"]
    fittable_minvals = [.01,.01,.01,.01,0.5,-8]
    fittable_maxvals = [8,8,8,8,1.5,8]
    fixedC = [1,1,1,1,1,0]
    freeP = [1,1,1,1,1,1]

    def get_drift(self, conditions, **kwargs):
        visContrast = np.abs(conditions["visDiff"])
        visSide = np.sign(conditions["visDiff"])
        audSide = np.sign(conditions["audDiff"])
        
        myDrift = (self.aR * (audSide>0) - self.aL * (audSide<0) + 
                   self.vR * (visSide>0) * (visContrast**self.gamma) - self.vL * (visSide<0) * (visContrast**self.gamma) +
                   self.b)

        return myDrift

class OverlayNonDecisionAV(pyddm.OverlayNonDecision):
    name = "Separate non-decision time for aud and vis components"
    required_parameters = ["nondectimeA", "nondectimeV"]
    required_conditions = ["audDiff",'visDiff'] # Side coded as 0=L or 1=R
    fittable_minvals = [.01,.01]
    fittable_maxvals = [.4,.4]
    fixedC = [.3,.3]
    freeP = [1,1]    

    def get_nondecision_time(self, conditions):
        isV = np.abs(conditions["visDiff"])!=0
        isA = np.abs(conditions["visDiff"])!=0

        if isA and not isV: 
            nondectime = self.nondectimeA
        elif isV and not isA:
            nondectime = self.nondectimeV
        else: 
            nondectime = np.mean([self.nondectimeV,self.nondectimeA])

        return nondectime 

    
def get_default_noise():
    c = pyddm.NoiseConstant
    c.fittable_minvals = [.2]
    c.fittable_maxvals = [3]
    c.fixedC = [1]
    c.freeP = [1]
    return c

def get_default_bound():
    c = pyddm.BoundConstant
    c.fittable_minvals = [1]
    c.fittable_maxvals = [3]
    c.fixedC = [1]
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

def get_default_IC():
    c = pyddm.ICPoint
    c.fittable_minvals = [-.9]
    c.fittable_maxvals = [.9]
    c.fixedC = [0]
    c.freeP = [1]
    return c

def DriftAdditiveSplit_freeP_sets(which = 'starting_point',nondecType = 'OverlayNonDecisionAV'):
    
    if 'starting_point' in which: 
        freePs = {
            'drift': [0,0,0,0,0,0], 
            'noise': [0],
            'bound': [0],
            'nondectime':[0],
            'mixture':[0,0],
                'IC': [1]
        }
    
    elif 'constant_bias' in which:
        freePs = {
            'drift': [0,0,0,0,0,1], 
            'noise': [0],
            'bound': [0],
            'nondectime':[0],
            'mixture':[0,0],
                'IC': [0]
        }

    elif 'sensory_drift' in which:
        freePs = {
            'drift': [1,1,1,1,0,0], 
            'noise': [0],
            'bound': [0],
            'nondectime':[0],
            'mixture':[0,0],
                'IC': [0]
        }

    elif 'driftIC' in which:
        freePs = {
            'drift': [0,0,0,0,0,1], 
            'noise': [0],
            'bound': [0],
            'nondectime':[0],
            'mixture':[0,0],
                'IC': [1]
        }

    elif 'all' in which:
        freePs = {
            'drift': [1,1,1,1,0,1], 
            'noise': [0],
            'bound': [0],
            'nondectime':[0],
            'mixture':[0,0],
                'IC': [1]
        }
   
    if nondecType is not None:
        freePs['nondectime'] = [0,0]

    return freePs


