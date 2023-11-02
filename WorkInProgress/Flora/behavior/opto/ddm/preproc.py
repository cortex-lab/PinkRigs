"""
functions to preprocess the data

"""
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle
import pyddm
import pandas as pd

def preproc_ev(ev): 
    """
    function to preprocess the event structure coming out of the PinkRig pipeline to be suitable for pyddm

    Parameters:
    -----------

    ev: pd.df  
        specifies whether to calculate a new column called trainSet for holding out data 
        split occurs based on sklearn.StratifiedShuffleSplit
        Stratification occurs based on visStim, audStim and choice
    splitkwargs: dict
        input parameters for  sklearn.StratifiedShuffleSplit 

    """
    ev['visDiff'] = ev.stim_visDiff
    ev['audDiff'] = ev.stim_audDiff
    ev['visDiff'] = np.round(ev.visDiff/max(ev.visDiff),2) # aud is already normalised byt we also normalise vis
    ev["choice"] = (ev["response_direction"]-1).astype(int)
    ev['rt_thresh'] = ev.timeline_choiceThreshOn-ev.timeline_audPeriodOn
    ev['rt_laserThresh'] = ev.timeline_choiceThreshPostLaserOn-ev.block_laserStartTimes
    ev['stimulated_hemisphere'] = np.sign(ev.laser_power_signed)
    ev['RT'] = ev['rt_laserThresh']
    return ev

def cv_split(ev,**splitkwargs):
    """
    function to calculate a new column called trainSet for holding out data 
    split occurs based on sklearn.StratifiedShuffleSplit
    Stratification occurs based on visStim, audStim and choice

    Parameters: 
    -----------
    ev: pd.df  
    splitkwargs: dict
        input parameters for  sklearn.StratifiedShuffleSplit 
        
    """

    _,ev['groupID'], counts = np.unique(ev[['visDiff','audDiff','choice']],axis=0,return_inverse=True,return_counts=True)

    # in some datasets the count is 1 for certain groups -- that is not worth estimating so we throw this datapoint out
    singles = np.where(counts==1)[0]

    if singles.size>0:
        is_single_count = np.sum(np.concatenate([(ev.groupID==smallset).values.astype('int')[np.newaxis,:] for smallset in np.where(counts==1)[0]]),axis=0)
        is_single_count = is_single_count.astype('bool')
        ev=ev[~is_single_count]

    sss = StratifiedShuffleSplit(**splitkwargs)
    _,(train_idx,_) = sss.split(ev['RT'],ev[['choice','groupID']])
    ev['trainSet'] = False
    ev['trainSet'].iloc[train_idx]=True

    return ev

def save_pickle(mydict,path):
    with open(path.__str__(), 'wb') as f:
        pickle.dump(mydict,f,pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path.__str__(),'rb') as f:
        obj = pickle.load(f)
    return obj

def resample_model(model,sample_path=None,split=True):
    """
    resampling of a pyddm model

    # we just need to read in how many choices were made in the actial sample with that condition
    """

    if sample_path: 
        data = read_pickle(sample_path)
        actual_aud_azimuths = np.sort(data.to_pandas_dataframe().audDiff.unique())
        actual_vis_contrasts = np.sort(data.to_pandas_dataframe().visDiff.unique())
    else:
        actual_aud_azimuths = [-60,0,60]
        actual_vis_contrasts = [-1,-.5,-.25,0,.25,.5,1]
    sample_df = []

    for isLaser in range(2):
        for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):
                curr_cond = {'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                sol = model.solve(conditions=curr_cond)

                if sample_path: 
                    d_set = data.subset(audDiff=a,visDiff=v,is_laserTrial=isLaser)
                    n_samples = d_set.choice_upper.size  + d_set.choice_lower.size
                else: 
                    n_samples = 50

                sample = sol.resample(n_samples)
                sample_df.append(sample.to_pandas_dataframe(drop_undecided=True))

    sample_df = pd.concat(sample_df)

    if split:
        Block = cv_split(sample_df,n_splits=2,test_size=.2,random_state=0)
        train = pyddm.Sample.from_pandas_dataframe(Block[Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
        test = pyddm.Sample.from_pandas_dataframe(Block[~Block.trainSet], rt_column_name="RT", choice_column_name="choice", choice_names =  ("Right", "Left"))
    else: 
        
        train = pyddm.Sample.from_pandas_dataframe(sample_df,
            rt_column_name='RT',
            choice_column_name='choice',
            choice_names =  ("Right", "Left"))
        test=None

    
    return train,test