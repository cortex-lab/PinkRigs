import pyddm 
import time
import pandas as pd 
import numpy as np


from preproc import save_pickle,preproc_ev,cv_split
from pathlib import Path



def write_samples():
    # input data paths
    data_path = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data')
    animal_paths = list(data_path.glob('*10mW*.csv'))

    # output data paths
    savepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\samples')
    savetrain = savepath / 'train'
    savetest = savepath / 'test'
    saveall = savepath / 'all'

    savetrain.mkdir(parents=True,exist_ok=True)
    savetest.mkdir(parents=True,exist_ok=True)
    saveall.mkdir(parents=True,exist_ok=True)

    for animal_path in animal_paths:

        ev = pd.read_csv(animal_path) 
        ev = preproc_ev(ev)

        s = animal_path.stem
        Block = ev[~np.isnan(ev.rt_laserThresh)]
        Sample_ = pyddm.Sample.from_pandas_dataframe(Block, rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        save_pickle(Sample_,saveall / ('%s_Sample_all.pickle' % s))

        Block = cv_split(Block,n_splits=2,test_size=.2,random_state=0)
        Sample_train = pyddm.Sample.from_pandas_dataframe(Block[Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        save_pickle(Sample_train,savetrain / ('%s_Sample_train.pickle' % s))
        Sample_test = pyddm.Sample.from_pandas_dataframe(Block[~Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        save_pickle(Sample_test,savetest / ('%s_Sample_test.pickle' % s))


if __name__ == "__main__":  
   write_samples() 

