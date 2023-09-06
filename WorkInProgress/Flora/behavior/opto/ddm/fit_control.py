# %% 

# prepare the data into correct pandas format
import sys
import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,concatenate_events

my_subject = ['AV030','AV025','AV034']
recordings = load_data(
    subject = my_subject,
    expDate = '2021-05-02:2023-09-20',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )   

ev = concatenate_events(recordings,filter_type='final_stage')

ev_ = pd.DataFrame.from_dict(ev)

ev_ = ev_.dropna(subset=['rt'])
# %%
# for now we are not simulating nogos
import pyddm 
import pyddm.plot
from models import DriftAdditive


sample = pyddm.Sample.from_pandas_dataframe(ev_[ev_.subject=='AV025'], rt_column_name="rt", correct_column_name="response_direction")


m = pyddm.Model(drift=DriftAdditive(aud_coef=pyddm.Fittable(minval=0, maxval=5),
                                    vis_coef=pyddm.Fittable(minval=0, maxval=5)),
                noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=10)),
                bound=pyddm.BoundConstant(B=1),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.3)),
                    pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=0, maxval=.4),
                                                    rate=1),
                    ]),
                IC=pyddm.ICPoint(x0=pyddm.Fittable(minval=0, maxval=1)),
                dt=.001, dx=.01, T_dur=4)

pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": [-1, 0, 1], "visDiff": np.sort(ev_.visDiff.unique())})


# %% 
# or stop trying to fool around with the gui and just fit it.... because there are so many params that intuitively it is not clear to me how to fit it 

#pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossRobustLikelihood, verbose=False)
# %%
