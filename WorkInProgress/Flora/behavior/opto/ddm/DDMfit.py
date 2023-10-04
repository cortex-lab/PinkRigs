import pyddm 
import time
import pandas as pd 
import numpy as np


from fitting import get_parameters
from model_components import DriftAdditiveSplit_freeP_sets
from preproc import save_pickle,read_pickle,preproc_ev,cv_split
from pathlib import Path



def trainDDMs(refit=False):
    pyddm.set_N_cpus(16) # max is 16 core on my computer 

    # input data paths
    data_path = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\Data\bilateral')
    animal_paths = list(data_path.glob('*10mW*.csv'))

    # output data paths
    basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\bilateral')
    model_name = 'DriftAdditiveSplit'
    savepath = basepath / model_name
    savepath.mkdir(parents=True,exist_ok=True)
    nondecType = None

    fit_params = get_parameters(driftType=model_name,nondectimeType=None,model=None)

    refit_freePs = [
        'starting_point', 
        'constant_bias',
        'sensory_drift', 
        'driftIC', 
        'all'
    ]

    for animal_path in animal_paths:

        ev = pd.read_csv(animal_path) 
        ev = preproc_ev(ev)

        s = animal_path.stem

        ctrlBlock = ev[~np.isnan(ev.rt_laserThresh) & ~ev.is_laserTrial.astype('bool')] 
        ctrlSample = pyddm.Sample.from_pandas_dataframe(ctrlBlock, rt_column_name="rt_laserThresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))
        save_pickle(ctrlSample,savepath / ('%s_CtrlSample.pickle' % s))


        optoBlock = ev[~np.isnan(ev.rt_laserThresh) & ev.is_laserTrial.astype('bool')] 
        # split the optoBlock to train and test test
        optoBlock = cv_split(optoBlock,n_splits=2,test_size=.2,random_state=0)

        optoSample_train = pyddm.Sample.from_pandas_dataframe(optoBlock[optoBlock.trainSet], rt_column_name="rt_laserThresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))
        save_pickle(optoSample_train,savepath / ('%s_OptoSample_train.pickle' % s))

        optoSample_test = pyddm.Sample.from_pandas_dataframe(optoBlock[~optoBlock.trainSet], rt_column_name="rt_laserThresh", choice_column_name="response_direction_fixed", choice_names =  ("Right", "Left"))
        save_pickle(optoSample_test,savepath / ('%s_OptoSample_test.pickle' % s))

        try:
            t0 = time.time()
            currmodel_path  = savepath / ('%s_CtrlModel.pickle' % s)
            if refit or not (currmodel_path.is_file()):               
                print('control fit %s...' %s )
                m = pyddm.Model(**fit_params)
                pyddm.fit_adjust_model(model=m, sample=ctrlSample, lossfunction=pyddm.LossLikelihood, verbose=False)         
                print('%s successfully fit,saving...' % s)
                save_pickle(m,savepath / currmodel_path)
                print('time to fit:%.2d s' % (time.time()-t0))
            else: 
                # the control fit is crucial for the rest of the fitting so need to read in the model
                m = read_pickle(currmodel_path)   
            
            #optofit_loss = pyddm.get_model_loss(model=m,sample=optoSample) 

            # reistante models with various paramters fixed vs not fixed
            for set in refit_freePs:  
                currmodel_path  = savepath / ('%s_OptoModel_%s.pickle' % (s,set)) 
                if refit or not (currmodel_path.is_file()):
                    print('fitting opto model with changing %s ...' % set)
                    freePs = DriftAdditiveSplit_freeP_sets(set,nondecType=nondecType)
                    opto_params = get_parameters(model=m,freePs=freePs)
                    o = pyddm.Model(**opto_params)
                    pyddm.fit_adjust_model(model=o, sample=optoSample_train, lossfunction=pyddm.LossLikelihood, verbose=False)
                    save_pickle(o,savepath / currmodel_path)

        except FloatingPointError:                  
            print('Constant floating point error with %s. Probably best to reexamine parameters?')



if __name__ == "__main__":  
   trainDDMs(refit=False) 

