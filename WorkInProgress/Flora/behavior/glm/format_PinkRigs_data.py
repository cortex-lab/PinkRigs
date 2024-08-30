# %%
# this 
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import simplify_recdat
from loaders import load_params
from predChoice import format_av_trials
# this queries the csv for possible recordings 

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.loaders import call_neural_dat


my_ROI = 'MOs'
paramset_name = 'choice'
params = load_params(paramset=paramset_name)
savepath = Path(r'D:\LogRegression\%s_%s' % (my_ROI,paramset_name))
selected_sessions = call_neural_dat(
                            subject_set='forebrain',
                            dataset_type='active',
                             spikeToInclde=True,
                             camToInclude=False,
                             recompute_data_selection=True,
                             unwrap_probes= False,
                             merge_probes=True,
                             filter_unique_shank_positions = False,
                             region_selection={'region_name':my_ROI,
                                                'framework':'Beryl',
                                                'min_fraction':20,
                                                'goodOnly':True,
                                                'min_spike_num':300},
                             min_rt=params['post_time'],
                             analysis_folder = savepath
                             )

# %%
for _,rec in selected_sessions.iterrows():
    ev,spk,clusInfo,_,cam = simplify_recdat(rec,probe='probe')
    goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym==my_ROI)]._av_IDs.values
    selected_sessions['neuronNo'] = goodclusIDs.size

#
# Group by 'subject' and aggregate
result = selected_sessions.groupby('subject').agg(
    session_count=('subject', 'size'),  # Count the number of rows per subject
    neuron_sum=('neuronNo', 'sum')  # Sum the 'neuronNo' values per subject
).reset_index()

print(result)

#%%
#rec = selected_sessions.iloc[0]
savepath = savepath / 'formatted'

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

    savepath = Path(savepath)
    savepath.mkdir(parents=False,exist_ok=True)

    sessname = '{subject}_{expDate}_{expNum}.csv'.format(**rec)

    trials.to_csv((savepath / sessname),index=False)
# %%
