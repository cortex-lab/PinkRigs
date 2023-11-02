import pyddm 
import time
import pandas as pd 
import numpy as np


from fitting import get_parameters
from model_components import get_freeP_sets
from preproc import save_pickle,preproc_ev,cv_split
from pathlib import Path



def trainDDMs(refit=False):
    pyddm.set_N_cpus(16) # max is 16 core on my computer 

    # input data paths
    data_path = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data')
    animal_paths = list(data_path.glob('*10mW*.csv'))

    # output data paths
    basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
    drift_class = 'DriftAdditiveOpto'

    savepath = basepath / drift_class
    savepath.mkdir(parents=True,exist_ok=True)

    # freeP_sets = [
    #     'ctrl',
    #     'drift_bias'
    #     # # 'sensory_drift',
    #     # 'starting_point'
    #     # 'mixture',
    #     # 'nondectime',
    #     # 'all'
    # ]

    freeP_sets = [
        'all',
        'l_a',
        'l_v', 
        'l_aS',
        'l_vS',
        'l_gamma',
        'l_b',
        'l_d_aR',
        'l_d_aL',
        'l_d_vR',
        'l_d_vL',
        'l_d_b',
        'l_d_nondectime', 
        'l_d_mixturecoef',
        'l_d_x0'
    ] # (15 models currently)


    for animal_path in animal_paths:

        ev = pd.read_csv(animal_path) 
        ev = preproc_ev(ev)

        s = animal_path.stem

        Block = cv_split(ev[~np.isnan(ev.rt_laserThresh)],n_splits=2,test_size=.2,random_state=0)
        Sample_train = pyddm.Sample.from_pandas_dataframe(Block[Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        save_pickle(Sample_train,savepath / ('%s_Sample_train.pickle' % s))
        Sample_test = pyddm.Sample.from_pandas_dataframe(Block[~Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        save_pickle(Sample_test,savepath / ('%s_Sample_test.pickle' % s))

        try:
            t0 = time.time()
            for set in freeP_sets:  
                currmodel_path  = savepath / ('%s_Model_%s.pickle' % (s,set)) 
                if refit or not (currmodel_path.is_file()):  
                    freePs = get_freeP_sets(set)
                    fit_params = get_parameters(freePs=freePs)             
                    print('%s fit %s...' %(set,s))
                    m = pyddm.Model(**fit_params)
                    pyddm.fit_adjust_model(model=m, sample=Sample_train, lossfunction=pyddm.LossLikelihood, verbose=False)         
                    print('%s successfully fit,saving...' % s)
                    save_pickle(m,savepath / currmodel_path)
                    print('time to fit:%.2d s' % (time.time()-t0))    
            
        except FloatingPointError:                  
            print('Constant floating point error with %s. Probably best to reexamine parameters?')



if __name__ == "__main__":  
   trainDDMs(refit=False) 

