
import pyddm 
import time
import pandas as pd 
import numpy as np
from models import DriftAdditive,DriftAdditiveSplit,get_default_parameters
from pathlib import Path
import pickle



def fitDDMs(refit=False,max_attempts = 3):
    pyddm.set_N_cpus(16) # max is 16 core on my computer 

    # input data paths
    data_path = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\Data')
    animal_paths = list(data_path.glob('*.csv'))

    # output data paths
    basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto')
    model_name = 'DriftAdditiveSplit'
    savepath = basepath / model_name
    savepath.mkdir(parents=True,exist_ok=True)
    params = get_default_parameters(modelType=model_name)


    for animal_path in animal_paths:

        ev = pd.read_csv(animal_path) 
        ev['visDiff'] = ev.stim_visDiff
        ev['audDiff'] = ev.stim_audDiff
        ev['visDiff'] = np.round(ev.visDiff/max(ev.visDiff),2) # aud is already normalised byt we also normalise vis
        ev["response_direction_fixed"] = (ev["response_direction"]-1).astype(int)
        ev['rt_thresh'] = ev.timeline_choiceThreshOn-ev.timeline_audPeriodOn
        ev['rt_laserThresh'] = ev.timeline_choiceThreshPostLaserOn-ev.block_laserStartTimes
        
        
        ev = ev[~np.isnan(ev.rt_laserThresh) & ~ev.is_laserTrial.astype('bool')] 

        # create 

        s = animal_path.stem

        model_path = (savepath / ('%s_Model.pickle' % s))
        sample_path = (savepath / ('%s_Sample.pickle' % s))

        if refit or not (model_path.is_file() & sample_path.is_file()):
        
            sample = pyddm.Sample.from_pandas_dataframe(ev, rt_column_name="rt_laserThresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))

            m = pyddm.Model(**params)
            
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