import pyddm 
import time
import itertools
import os 
import psutil
import sys

from fitting import get_parameters
from model_components import get_freeP_sets
from preproc import save_pickle,read_pickle
from pathlib import Path

# for the parallel 
print('CSOLVE CORRECRLY LOADED:', pyddm.model.HAS_CSOLVE)

def trainDDMs(rank=1):
    """
    primarily a parallelisable finction to fit ddms
    
    """
    rank = float(rank)
    #pyddm.set_N_cpus(16) # set to 6 core/cpu
    # input data paths

    fit_resampled_data = False 
    mycwd = Path(os.getcwd())
    
    if fit_resampled_data:
         data_path = mycwd / 'resample_data/train'

    else:
        data_path = mycwd / 'train'
    data_path = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto\Data\forMyriad\samples\train')
    animal_paths = list(data_path.glob('*_10mW_*.pickle')) # 11 files 
    #animal_paths = list(data_path.glob('*g_both*.pickle')) # 11 files 
    # output data paths
    #basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto')
    if fit_resampled_data:
        drift_class = 'resampled_fits'
    else:
        drift_class = 'DriftAdditiveOpto'

    savepath = mycwd / drift_class
    savepath.mkdir(parents=True,exist_ok=True)


    freeP_sets = [
        #'ctrl',
        #'all',
        # 'l_aS',
        # 'l_vS',
        # 'l_gamma',
        # 'l_b',
        # 'l_d_aR',
        # 'l_d_aL',
        # 'l_d_vR',
        # 'l_d_vL',
        # 'l_d_b',
        # 'l_d_nondectime', 
        # 'l_d_mixturecoef',
        # 'l_d_x0',
        # 'g_d_x0',
        # 'g_d_b',
        # 'g_both',
        'g_boundx0'

    ] 
    # loop over animals x products and send the animals x models 

    print(len(animal_paths)*len(freeP_sets), 'models to fit...')

    for i,(animal_path,set) in enumerate(itertools.product(animal_paths,freeP_sets)):
        s = animal_path.stem
        currmodel_path  = savepath / ('%s_Model_%s.pickle' % (s,set)) 
        if (i==(rank-1)) and not currmodel_path.is_file(): 
            # preproc
            Sample_train = read_pickle(animal_path) 

            t0 = time.time()
            freePs = get_freeP_sets(set)
            fit_params = get_parameters(freePs=freePs)  
            m = pyddm.Model(**fit_params)
            pyddm.fit_adjust_model(model=m, sample=Sample_train, lossfunction=pyddm.LossLikelihood, verbose=False)         
            save_pickle(m,savepath / currmodel_path)
            print('%s fit %s...' %(set,s))
            print('time to fit:%.2d s' % (time.time()-t0))    
            print('RAM Used (GB):', (psutil.virtual_memory()[3]/1000000000))
            print('memory used (GB):', (psutil.Process().memory_info().rss / (1024 * 1024 *1000)))


            


if __name__ == "__main__":  
   trainDDMs(rank=sys.argv[1]) 
   # trainDDMs(rank=1) 
