"""
model recovery generating data with various parameters
and compare
"""

# %% 
from model_components import DriftAdditiveOpto,OverlayExponentialMixtureOpto,OverlayNonDecisionOpto,ICPointOpto
import pyddm
import numpy as np

a = np.arange(0,8,10)
v = np.arange(0,8,10)


fit_params = {'drift': DriftAdditiveOpto(a=1, v=1, aS=1, vS=1, gamma=1, b=0, 
                                         d_aR=0, d_aL=0, d_vR=0, d_vL=0, d_b=0),
              'noise': pyddm.NoiseConstant(noise=1),
              'bound': pyddm.BoundConstant(B=1),
              'overlay': pyddm.OverlayChain(overlays=[OverlayNonDecisionOpto(nondectime=0.3, d_nondectimeOpto=0), 
                                   OverlayExponentialMixtureOpto(pmixturecoef=0.2, rate=1, d_pmixturecoef=0)]),
                'IC': ICPointOpto(x0=0, d_x0=0),
                'dt': 0.001,
                'dx': 0.001,
                'T_dur': 2,
                  'choice_names': ('Right', 'Left')}




# %%
