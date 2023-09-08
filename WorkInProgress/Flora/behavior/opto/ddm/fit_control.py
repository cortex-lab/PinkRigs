# %% 

#prepare the data into correct pandas format
# import sys
# import numpy as np
# import pandas as pd
# from itertools import compress
# import matplotlib.pyplot as plt

# sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
# from Admin.csv_queryExp import load_data,concatenate_events

# my_subject = ['AV030','AV025','AV034']
# recordings = load_data(
#     subject = my_subject,
#     expDate = '2021-05-02:2023-09-20',
#     expDef = 'multiSpaceWorld_checker_training',
#     checkEvents = '1', 
#     data_name_dict={'events':{'_av_trials':'table'}}
#     )   

# ev = concatenate_events(recordings,filter_type='final_stage')

# ev_ = pd.DataFrame.from_dict(ev)

# ev_ = ev_.dropna(subset=['rt'])
# %%
# for now we are not simulating nogos
import numpy as np
import pandas as pd
ev_ = pd.read_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv')

ev_["response_direction_fixed"] = (ev_["response_direction"]-1).astype(int)
import pyddm 
import pyddm.plot
#%%
class DriftAdditive(pyddm.Drift):
    name = "additive drift"
    required_parameters = ["aud_coef", "vis_coef",'contrast_power']
    required_conditions = ["audDiff", "visDiff"]
    def get_drift(self, conditions, **kwargs):
        return (self.aud_coef * float(conditions["audDiff"]) + self.vis_coef * (float(conditions["visDiff"])**self.contrast_power)).real
    

#%%
sample = pyddm.Sample.from_pandas_dataframe(ev_[ev_.subject=='AV025'], rt_column_name="rt", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

m = pyddm.Model(drift=DriftAdditive(aud_coef=pyddm.Fittable(minval=0, maxval=50),
                                    vis_coef=pyddm.Fittable(minval=0, maxval=50), contrast_power = pyddm.Fittable(minval=0, maxval=10)),
                noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=20)),
                bound=pyddm.BoundConstant(B=1),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.3)),
                    pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=0, maxval=.6),
                    rate=pyddm.Fittable(minval=0, maxval=10)),
                    ]),
                IC=pyddm.ICPoint(x0=pyddm.Fittable(minval=-.9, maxval=.9)),
                dt=.001, dx=.01, T_dur=4,choice_names = ('Right','Left'))

pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": [-1, 0, 1], "visDiff": np.sort(ev_.visDiff.unique())})

#"pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)

# %% 
# or stop trying to fool around with the gui and just fit it.... because there are so many params that intuitively it is not clear to me how to fit it 

#pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
# %%
# fit and then assess how we predict the psychometric and the chronometric


