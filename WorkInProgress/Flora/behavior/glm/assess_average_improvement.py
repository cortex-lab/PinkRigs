# %%

# fitting with the average activity and assess whether fit is improving

import sys
import numpy as np

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat
from Analysis.neural.utils.data_manager import load_cluster_info


ephys_dict = {'spikes':'all','clusters':'all'}
recordings = load_data(data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict,'events': {'_av_trials': 'table'}},
                        subject = ['AV025','AV030','AV034'],expDate='postImplant',
                        expDef='multiSpaceWorld',
                        checkEvents='1',
                        checkSpikes='1',
                        unwrap_independent_probes=True,
                        region_selection={'region_name':'SC','min_fraction':.3})
n_trials  = np.array([((rec.events._av_trials.response_direction>0) & (rec.events._av_trials.is_validTrial)).sum() for _,rec in recordings.iterrows()])

recordings = recordings.iloc[n_trials>100]

# %%
from predChoice import format_av_trials,glmFit

def compare_with_neural_average(rec,to_plot=False):
    # preselect clusters based on quality metrics 
    ev,spk,_,_,_ = simplify_recdat(rec,probe='probe')
    clusInfo = load_cluster_info(rec,probe='probe')

    # could build a helper for this
    from Processing.pyhist.helpers.regions import BrainRegions
    from Analysis.neural.utils.spike_dat import bombcell_sort_units
    br = BrainRegions()
    bc_class = bombcell_sort_units(clusInfo)
    clusInfo['is_good'] = bc_class=='good'
    clusInfo.brainLocationAcronyms_ccf_2017[clusInfo.brainLocationAcronyms_ccf_2017=='unregistered'] = 'void' # this is just so that the berylacronymconversion does something good
    clusInfo['BerylAcronym'] = br.acronym2acronym(clusInfo.brainLocationAcronyms_ccf_2017, mapping='Beryl')

    goodclusIDs = clusInfo[(clusInfo.is_good)&(clusInfo.BerylAcronym=='SCm')]._av_IDs.values

    #goodclusIDs = [158,228,122]


    event_type = 'timeline_audPeriodOn'
    trials = format_av_trials(ev,spikes=spk,nID=goodclusIDs,single_average=True,t=0.15,onset_time=event_type)


# fit with and without the average and compare...

    logLiks = []
    if to_plot:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(2,3,figsize=(15,8))
    for i in range(2):
        glm = glmFit(trials.iloc[:,:(3+i)],model_type='AVSplit',fixed_parameters = [0,0,0,0,0,0])
        glm.fitCV(n_splits=2,test_size=0.5)
        logLiks.append(glm.model.LogLik)

        if to_plot:
            glm.visualise(yscale='log',ax = ax[0,i])
            ax[0,i].set_title('LogLik: %.2f' % glm.model.LogLik)
            glm.plotPrediction(yscale='sig',ax=ax[1,i])
            ax[1,2].hist(glm.model.get_logOdds(glm.conditions,glm.model.allParams),alpha=0.5,range=(-8,8)) 
   
        
            ax[1,2].set_xlabel('LogOdds')
            ax[1,2].set_ylabel('# trials')

            from Analysis.pyutils.plotting import off_axes
            off_axes(ax[0,2])


    return np.array(logLiks)

# %%


lls = [compare_with_neural_average(rec,to_plot=True)[np.newaxis,:] for _,rec in recordings.iterrows()]
lls = np.concatenate(lls)
# %%
import matplotlib.pyplot as plt

plt.hist(lls[:,0]-lls[:,1],range=(-.1,.1))
# %%
