# %%
import pyddm
import pyddm.plot
import pandas
import numpy as np 

# model
from models import DriftAdditive

aud_coef = 4
vis_coef = 3
noise=1
bound = .3
nondecttime=.5

m = pyddm.Model(drift=DriftAdditive(aud_coef=aud_coef,
                                        vis_coef=vis_coef),
                noise=pyddm.NoiseConstant(noise=noise),
                bound=pyddm.BoundConstant(B=bound),
                overlay=pyddm.OverlayNonDecision(nondectime=nondecttime),
                dt=.01, dx=.01, T_dur=4)


# %%
# plot psychometric  
import itertools
import matplotlib.pyplot as plt
aud_azimuths  = np.linspace(-1,1,3)
vis_contrasts = np.linspace(-1,1,40)

psychometric,a,v = zip(*[[m.solve(conditions={"audDiff": a, "visDiff": v}).prob('correct'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])

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
