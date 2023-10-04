import numpy as np
import pyddm 


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
    
    