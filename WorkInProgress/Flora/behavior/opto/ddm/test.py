

import pyddm 
import pandas as pd 
import numpy as np
from models import DriftAdditive
from pathlib import Path
import pickle


def fitDDMs():
    pyddm.set_N_cpus(16) # max is 16 core on my computer 
    ev = pd.read_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv') # batch data
    savepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm') # where we save out the models/fitted parameters

    ev['visDiff'] = np.round(ev.visDiff/max(ev.visDiff),2) 

    subjects = ev.subject.unique()

    for s in subjects: 

        sample = pyddm.Sample.from_pandas_dataframe(ev[ev.subject==s], rt_column_name="rt_thresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

        m = pyddm.Model(drift=DriftAdditive(aud_coef=pyddm.Fittable(minval=0, maxval=15),
                                            vis_coef=pyddm.Fittable(minval=0, maxval=15)),# contrast_power = pyddm.Fittable(minval=0, maxval=10)),
                        noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=5)),
                        bound=pyddm.BoundConstant(B=1),
                        overlay=pyddm.OverlayChain(overlays=[
                            pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.7)),
                            pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=0, maxval=.6),
                            rate=pyddm.Fittable(minval=.8, maxval=2))]),
                        IC=pyddm.ICPoint(x0=pyddm.Fittable(minval=-.9, maxval=.9)),
                        dt=.001, dx=.001, T_dur=4,choice_names = ('Right','Left'))

        print('fitting data from %s' % s)
        fitted = pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossLikelihood, verbose=False)

        # savepath
        print('%s successfully fit,saving...' % s)
        current_savepath = (savepath / ('%s_%s.pkl' % (s,'AdditiveDrift')))
        file = open(current_savepath.__str__(),'w')
        pickle.dump(m, file)
        print('saved.')


if __name__ == "__main__":  
   fitDDMs() 