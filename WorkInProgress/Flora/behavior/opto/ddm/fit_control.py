# %%
# for now we are not simulating nogos
import numpy as np
import pandas as pd

# set pydd
import pyddm 
import pyddm.plot
#pyddm.set_N_cpus(4)
# read in data K
ev_ = pd.read_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv')

#ev_['visDiff'] = np.round(ev_.visDiff/max(ev_.visDiff),2) # aud is already normalised byt we also normalise vis
#%%

def urgency_gain(t, gain_start, gain_slope):
    return gain_start + t*gain_slope


class DriftAdditive(pyddm.Drift):
    name = "additive drift"
    required_parameters = ["aud_coef", "vis_coef"]
    required_conditions = ["audDiff", "visDiff"]
    def get_drift(self, conditions, **kwargs):
        return (self.aud_coef * conditions["audDiff"] + self.vis_coef * (conditions["visDiff"])) #**self.contrast_power)).real
    
class DriftUrgencyGain(pyddm.Drift):
    name = "drift rate with an urgency function"
    required_parameters = ["aud_coef", "vis_coef", "gain_start", "gain_slope"]
    required_conditions = ["audDiff", "visDiff"]
    def get_drift(self,conditions, t, **kwargs):
        return (self.aud_coef * float(conditions["audDiff"]) + self.vis_coef * (float(conditions["visDiff"]))) * urgency_gain(t, self.gain_start, self.gain_slope)
    

#%%
sample = pyddm.Sample.from_pandas_dataframe(ev_[ev_.subject=='AV030'], rt_column_name="rt_thresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

m = pyddm.Model(drift=DriftAdditive(aud_coef=pyddm.Fittable(minval=0, maxval=200),
                                    vis_coef=pyddm.Fittable(minval=0, maxval=200)),# contrast_power = pyddm.Fittable(minval=0, maxval=10)),
                noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=10)),
                bound=pyddm.BoundConstant(B=1),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.5)),
                    pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=0, maxval=.6),
                    rate=1),
                    ]),
                IC=pyddm.ICPoint(x0=pyddm.Fittable(minval=-.9, maxval=.9)),
                dt=.001, dx=.001, T_dur=4,choice_names = ('Right','Left'))


pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": [-1, 0, 1], "visDiff": np.sort(ev_.visDiff.unique())})

import time
t0=time.time()
pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossLikelihood, verbose=False)
print(time.time()-t0)
# %% 

pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": [-1, 0, 1], "visDiff": np.sort(ev_.visDiff.unique())})

# or stop trying to fool around with the gui and just fit it.... because there are so many params that intuitively it is not clear to me how to fit it 

#pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
# %%
# predict the psychometric and the chronometric

import itertools
import matplotlib.pyplot as plt
aud_azimuths  = np.linspace(-1,1,3)
vis_contrasts = np.linspace(-1,1,40)

psychometric,a,v = zip(*[[m.solve(conditions={"audDiff": a, "visDiff": v}).prob('Right'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])

psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 


psychometric_log = np.log10(psychometric/(1-psychometric))
colors = ['b','k','r'] # for -1,0,1 aud

[plt.plot(vis_contrasts,p,color=c) for p,c in zip(psychometric,colors)]


# %% 
# chronometric
chronometric = ([m.solve(conditions={"audDiff": a, "visDiff": v}).mean_decision_time() for a,v in itertools.product(aud_azimuths,vis_contrasts)])
chronometric = np.reshape(np.array(chronometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 

[plt.plot(vis_contrasts,p,color=c) for p,c in zip(chronometric,colors)]

# %%
