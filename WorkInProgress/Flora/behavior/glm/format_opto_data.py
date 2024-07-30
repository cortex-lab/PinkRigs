#%%
from pathlib import Path
import pandas as pd
import numpy as np
#1 the opto data 
#2 the noGo data
basepath = Path(r'D:\LogRegression\opto\uni_all_nogo')
sessions = list(basepath.glob('*.csv'))

cpath = sessions[0]
# %%
ev = pd.read_csv(cpath)
# %%

maxV = np.max(np.abs(ev.stim_visDiff))
maxA = np.max(np.abs(ev.stim_audAzimuth))

trials = pd.DataFrame()
trials['visDiff']=ev.stim_visDiff/maxV
trials['audDiff']=ev.stim_audAzimuth/maxA
trials['choice'] = ev.response_direction-1
trials['feedback'] = ev.response_feedback
trials['opto'] = ev.is_laserTrial


# some further processing so that the predictors and visL/visR
trials['visR']=np.abs(trials.visDiff)*(trials.visDiff>0)
trials['visL']=np.abs(trials.visDiff)*(trials.visDiff<0)
trials['audR']=(trials.audDiff>0).astype('int')
trials['audL']=(trials.audDiff<0).astype('int')

# opto predictors for each
trials['visR_opto'] = trials.visR * trials.opto
trials['visL_opto'] = trials.visL * trials.opto
trials['audR_opto'] = trials.audR * trials.opto
trials['audL_opto'] = trials.audL * trials.opto



trials['trialtype_id'] = trials.copy().groupby(['visDiff','audDiff']).ngroup()


    # make sure that each class of trials will have min 2 types for splitting
uniqueIDs,test_counts = np.unique(trials.trialtype_id,return_counts=True)

if (test_counts<2).any():
    print('In %s I am dropping some trial types...' % cpath)
    rare_trialtypes = uniqueIDs[test_counts<2]
    for rareID in rare_trialtypes:
        trials = trials[trials.trialtype_id!=rareID]
# %%

stim_predictors = ['visR','visL','audR','audL']

# opto predictors
all_predictors =  stim_predictors + neural_predictors

# pred_list = ['visR','visL','audR','audL','neuron_142','neuron_1350','neuron_1351','neuron_1335']

X = trials[all_predictors]
y = trials['choice']
stratifyIDs = trials['trialtype_id']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1,shuffle=True,stratify=stratifyIDs)

