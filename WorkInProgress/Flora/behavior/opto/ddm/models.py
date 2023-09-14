
import pyddm

class DriftAdditive(pyddm.Drift):
    name = "additive drift"
    required_parameters = ["aud_coef", "vis_coef"]
    required_conditions = ["audDiff", "visDiff"]
    def get_drift(self, conditions, **kwargs):
        return (self.aud_coef * conditions["audDiff"] + self.vis_coef * (conditions["visDiff"])) #**self.contrast_power)).real
    


