

import pyddm 
import time
import pandas as pd 
import numpy as np
from models import DriftAdditive,DriftAdditiveSplit,get_default_parameters
from pathlib import Path
import pickle



def fitDDMs(refit=False,max_attempts = 3):
    pyddm.set_N_cpus(16) # max is 16 core on my computer 
    ev = pd.read_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv') # batch data
    ev = ev[ev.is_validTrial]

    basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm') # where we save out the models/fitted parameters

    model_name = 'AdditiveDriftSplit'
    savepath = basepath / model_name
    savepath.mkdir(parents=True,exist_ok=True)

    ev['visDiff'] = np.round(ev.visDiff/max(ev.visDiff),2) 

    subjects = ev.subject.unique()
    params = get_default_parameters()

    for s in subjects: 

        model_path = (savepath / ('%s_Model.pickle' % s))
        sample_path = (savepath / ('%s_Sample.pickle' % s))

        if refit or not (model_path.is_file() & sample_path.is_file()):
        
            sample = pyddm.Sample.from_pandas_dataframe(ev[ev.subject==s], rt_column_name="rt_thresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

            if 'AdditiveDriftSplit' in model_name:
                m = pyddm.Model(drift=DriftAdditiveSplit(aud_coef_right=pyddm.Fittable(minval=.01, maxval=8),
                                                         aud_coef_left=pyddm.Fittable(minval=.01, maxval=8),
                                                        vis_coef_right=pyddm.Fittable(minval=.01, maxval=8),
                                                        vis_coef_left=pyddm.Fittable(minval=.01, maxval=8),
                                                        contrast_power = pyddm.Fittable(minval=.02, maxval=4)), **params)
            else:    
                m = pyddm.Model(drift=DriftAdditive(aud_coef=pyddm.Fittable(minval=.01, maxval=8),
                                                    vis_coef=pyddm.Fittable(minval=.01, maxval=8),
                                                    contrast_power = pyddm.Fittable(minval=.02, maxval=4)), **params)

            
            print('fitting data from %s with model %s' % (s,model_name))
            t0 = time.time()
            attempts = 0 
            while attempts<=max_attempts:
                try: 
                    print('fitting attempt %.0d' % attempts)
                    pyddm.fit_adjust_model(model=m, sample=sample, lossfunction=pyddm.LossLikelihood, verbose=False)
                                    # savepath
                    print('%s successfully fit,saving...' % s)
                    print('time to fit:%.2d s' % (time.time()-t0))

                    # save the model
                    with open(model_path.__str__(), 'wb') as f:
                        pickle.dump(m, f,pickle.HIGHEST_PROTOCOL)

                    with open(sample_path.__str__(), 'wb') as f:
                        pickle.dump(sample, f,pickle.HIGHEST_PROTOCOL)

                    print('saved.')
                    attempts=max_attempts+1
                except FloatingPointError:                    
                    if attempts==max_attempts:
                        print('Constant floating point error with %s. Probably best to reexamine parameters?')
                    attempts+=1


if __name__ == "__main__":  
   fitDDMs(refit=True) 