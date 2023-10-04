# %%
# for now we are not simulating nogos
import numpy as np
import pandas as pd
from pathlib import Path

import pyddm 
import pyddm.plot
from models import DriftAdditiveSplitOpto,preproc_ev


basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\Data')
subject = 'AV036'
 

ev = pd.read_csv(basepath / ('%s.csv' % subject))
ev = preproc_ev(ev)
ev = ev[(~np.isnan(ev.rt_laserThresh) & ((ev.stimulated_hemisphere==-1) | np.isnan(ev.stimulated_hemisphere)))] 
ev['is_laserTrial'] = ev.is_laserTrial.astype('bool')
sample = pyddm.Sample.from_pandas_dataframe(ev, rt_column_name="rt_laserThresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

m = pyddm.Model(drift=DriftAdditiveSplitOpto(aud_coef_left=pyddm.Fittable(minval=.01, maxval=10),aud_coef_right=pyddm.Fittable(minval=.01, maxval=10), vis_coef_left = pyddm.Fittable(minval=.01, maxval=10), 
                                    vis_coef_right=pyddm.Fittable(minval=.01, maxval=10),contrast_power = pyddm.Fittable(minval=.2, maxval=2),laser_bias = pyddm.Fittable(minval=-.2, maxval=5)),
                noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=10)),
                bound=pyddm.BoundConstant(B=1),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=.01, maxval=.5)),
                    pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=.1, maxval=2),
                    rate=pyddm.Fittable(minval=.1, maxval=2)),
                    ]),
                IC=pyddm.ICPoint(x0=pyddm.Fittable(minval=-.9, maxval=.9)),
                dt=.001, dx=.001, T_dur=4,choice_names = ('Right','Left'))

actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))#
pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": actual_aud_azimuths, "visDiff": actual_vis_contrasts,"is_laserTrial":[False,True]})

# import time
# t0=time.time()
# pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossLikelihood, verbose=False)
# print(time.time()-t0)
# # %% 

# pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": [-1, 0, 1], "visDiff": np.sort(ev_.visDiff.unique())})

# # or stop trying to fool around with the gui and just fit it.... because there are so many params that intuitively it is not clear to me how to fit it 

# #pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
# # %%
# # predict the psychometric and the chronometric

# import itertools
# import matplotlib.pyplot as plt
# aud_azimuths  = np.linspace(-1,1,3)
# vis_contrasts = np.linspace(-1,1,40)

# psychometric,a,v = zip(*[[m.solve(conditions={"audDiff": a, "visDiff": v}).prob('Right'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])

# psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
# a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
# v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 


# psychometric_log = np.log10(psychometric/(1-psychometric))
# colors = ['b','k','r'] # for -1,0,1 aud

# [plt.plot(vis_contrasts,p,color=c) for p,c in zip(psychometric,colors)]


# # %% 
# # chronometric
# chronometric = ([m.solve(conditions={"audDiff": a, "visDiff": v}).mean_decision_time() for a,v in itertools.product(aud_azimuths,vis_contrasts)])
# chronometric = np.reshape(np.array(chronometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 

# [plt.plot(vis_contrasts,p,color=c) for p,c in zip(chronometric,colors)]

# %%
