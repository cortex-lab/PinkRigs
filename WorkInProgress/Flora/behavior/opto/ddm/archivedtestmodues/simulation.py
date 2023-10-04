# %%
import pyddm
import pyddm.plot
import pandas
import numpy as np 
import itertools
import matplotlib.pyplot as plt
from models import DriftAdditive

aud_coef = 2
vis_coef = 2
noise=1
bound = 1
contrast_power=1
x0 = [0,.1,.2,.3,.4,.5,.6]


m = pyddm.Model(drift=DriftAdditive(aud_coef=aud_coef,
                                        vis_coef=vis_coef,contrast_power =contrast_power),
                noise=pyddm.NoiseConstant(noise=noise),
                bound=pyddm.BoundConstant(B=bound),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=.3),
                    pyddm.OverlayExponentialMixture(pmixturecoef=0,
                    rate=1),
                    ]),
                IC=pyddm.ICPoint(x0=x0),
                dt=.01, dx=.01, T_dur=4)


# %%
# plot psychometric  

aud_azimuths  = np.linspace(-1,1,3)
vis_contrasts = np.linspace(-1,1,40)

psychometric,a,v = zip(*[[m.solve(conditions={"audDiff": a, "visDiff": v}).prob('correct'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])

psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 


psychometric_log = np.log10(psychometric/(1-psychometric))
colors = ['b','k','r'] # for -1,0,1 aud

fig,(ax,axc) = plt.subplots(1,2)
[ax.plot(vis_contrasts,p,color=c) for p,c in zip(psychometric_log,colors)]


# %% 
# chronometric
chronometric = ([m.solve(conditions={"audDiff": a, "visDiff": v}).mean_decision_time() for a,v in itertools.product(aud_azimuths,vis_contrasts)])
chronometric = np.reshape(np.array(chronometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 

[axc.plot(vis_contrasts,p,color=c) for p,c in zip(chronometric,colors)]

plt.show()
print('lah')
# %%
