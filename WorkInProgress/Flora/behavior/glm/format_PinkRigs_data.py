# %%
# this 
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import queryCSV,simplify_recdat
from loaders import load_rec_df,load_params
from predChoice import format_av_trials
# this queries the csv for possible recordings 

potential_sessions = queryCSV(
    subject = ['FT030','FT031','FT032','FT035','AV005','AV008','AV014','AV020','AV025','AV030','AV034'],
    expDate= 'postImplant',
    expDef='multiSpaceWorld',
    checkEvents='1',
    checkSpikes='1'
)

# we then load the acutal data and apply metrics to keep only certain sessions both on the behavioural performance
# and the neural data 

my_ROI = 'SCs'
paramset_name = 'poststim'

params = load_params(paramset=paramset_name)
# here the paramset is only used beecause the 
selected_sessions = load_rec_df(expList = potential_sessions,brain_area=my_ROI,paramset=paramset_name) 
#%%
#rec = selected_sessions.iloc[0]
for _,rec in selected_sessions.iterrows():
    ev,spk,clusInfo,_,cam = simplify_recdat(rec,probe='probe')
    goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym==my_ROI)]._av_IDs.values
    trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,cam=cam,**params)

    # some further processing so that the predictors and visL/visR
    trials['visR']=np.abs(trials.visDiff)*(trials.visDiff>0)
    trials['visL']=np.abs(trials.visDiff)*(trials.visDiff<0)
    trials['audR']=(trials.audDiff>0).astype('int')
    trials['audL']=(trials.audDiff<0).astype('int')
    trials['trialtype_id'] = trials.copy().groupby(['visDiff','audDiff']).ngroup()
    #

    # make sure that each class of trials will have min 2 types for splitting
    uniqueIDs,test_counts = np.unique(trials.trialtype_id,return_counts=True)

    if (test_counts<2).any():
        print('In %s I am dropping some trial types...' % rec.expFolder)
        rare_trialtypes = uniqueIDs[test_counts<2]
        for rareID in rare_trialtypes:
            trials = trials[trials.trialtype_id!=rareID]


    # now orgnaise the data for sklearn
    pred_list = [c for c,_ in trials.items() if 'neuron' in c]
    pred_list = ['visR','visL','audR','audL'] + pred_list

    # pred_list = ['visR','visL','audR','audL','neuron_142','neuron_1350','neuron_1351','neuron_1335']

    X = trials[pred_list]
    y = trials['choice']
    stratifyIDs = trials['trialtype_id']

    savepath = r'D:\LogRegression' 
    savepath = savepath + '/%s_%s' % (my_ROI,paramset_name)
    savepath = Path(savepath)
    savepath.mkdir(parents=False,exist_ok=True)

    sessname = '{subject}_{expDate}_{expNum}.csv'.format(**rec)

    trials.to_csv((savepath / sessname),index=False)
# %%
