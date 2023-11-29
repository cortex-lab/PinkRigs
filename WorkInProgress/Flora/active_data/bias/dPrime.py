import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 

from Admin.csv_queryExp import simplify_recdat, Bunch
from Analysis.pyutils.ev_dat import parse_events
from Analysis.neural.utils.spike_dat import get_binned_rasters
from Analysis.pyutils.plotting import my_rasterPSTH


def get_choicePrime(rec,t=.1, rt_min = 0.05, contrasts = 'blank',onset_names = ['timeline_audPeriodOn','timeline_choiceMoveOn'],plot_summary=True,plot_nrns=False):
    """
    function to calculate the average ChoicePrime per trial type 
    for each neuron in a recording

    Parameters:
    t: float 
     time before event onset to take as a bin
    contrasts:str/list
     determines what contrasts 
     options: blank_only 
              all
              low3  # with any number
     
    onset_names: list
     list of events keys prior to which we calculate the choicePrime in time t (audPeriod etc.)
    toPlot:bool
     whether to plot the train test against each other when calling this function
    """
    ev,spikes,clusters,_,_ = simplify_recdat(rec,probe='probe')

    try:
        unique_contrasts = np.sort(np.unique(ev.stim_visContrast))
        if isinstance(contrasts,str):
            if contrasts=='blank_only': 
                vC = [0]
            elif contrasts == 'all': 
                vC = list(unique_contrasts) 
            elif 'low' in contrasts: 
                howMany = int(contrasts[-1])
                vC = unique_contrasts[:howMany]

        ev,trial_types = parse_events(
                ev,vC,[ev.stim_audAmplitude[0]],[60,0,-60],[0],
                classify_choice_types=True,choice_types = [1,2], 
                rt_params = {'rt_min':rt_min,'rt_max':1.5}, 
                classify_rt = False, 
                min_trial = 2, 
                include_unisensory_aud = True, 
                include_unisensory_vis = False,add_crossval_idx_per_class=True
        )
        # pivot the table by choice pairs 
        grouping_indices = trial_types.groupby(by=['contrast','spl','vis_azimuths','aud_azimuths']).indices
        # regroup, and check whether each group has both choices; if not, then drop. 
        ev = ev.copy()
        ev.newIDs = np.empty(ev.is_blankTrial.size)*np.nan
        
        for idx,group in enumerate(grouping_indices.keys()):
            g_idx = grouping_indices[group]
            if len(g_idx)==2: # of there are two categories to calulate the ROC in between
                for g in g_idx:
                    ev.newIDs[ev.trial_type_IDs==g] = idx

        ev  = Bunch({k:ev[k][~np.isnan(ev.newIDs)] for k in ev.keys()})

        nTrials = ev.is_blankTrial.size
        print(nTrials,'trials are kept.')

        raster_kwargs = {
                'pre_time':t,
                'post_time':0, 
                'bin_size':t,
                'smoothing':0,
                'return_fr':True,
                'baseline_subtract': False, 
        }

        if nTrials>0:

            nOnsets = len(onset_names)
            if plot_summary: 
                fig1,axs1 = plt.subplots(1,nOnsets,figsize=(8*nOnsets,8),sharex=True,sharey=True)


            dvals={}
            for tidx,onset_time in enumerate(onset_names):
                t_on = ev[onset_time]
                r = get_binned_rasters(spikes.times,spikes.clusters,clusters._av_IDs,t_on,**raster_kwargs)
                resp_per_trial = r.rasters[:,:,0]

                # calculate a nrn x response matrix with L and R choice
                # up until this point, we do the same for LDA and for 
                cv,stimGroup = np.unique(ev.cv_set),np.unique(ev.newIDs)
                choicePrime_train,choicePrime_test = [],[]
                for idx,(c,s) in enumerate(list(itertools.product(cv,stimGroup))):
                    idx_L = (ev.cv_set==c) & (ev.newIDs==s) & (ev.choiceType==1)
                    idx_R = (ev.cv_set==c) & (ev.newIDs==s) & (ev.choiceType==2)

                    rl_diff = resp_per_trial[idx_R,:].mean(axis=0) - resp_per_trial[idx_L,:].mean(axis=0)
                    #within_group_std = ((resp_per_trial[idx_R,:]).std(axis=0) + (resp_per_trial[idx_L,:]).std(axis=0))/2
                    choicePrime = rl_diff#/(within_group_std)#+1e-5)
                    if c==cv[0]:
                        choicePrime_train.append(choicePrime[np.newaxis,:])
                    elif c==cv[1]:
                        choicePrime_test.append(choicePrime[np.newaxis,:])

                choicePrime_train,choicePrime_test = np.concatenate(choicePrime_train).T.mean(axis=1),np.concatenate(choicePrime_test).T.mean(axis=1) # nrn x stimGroup matrix indicating choicePrime
                
                dvals[onset_time] = Bunch({'train':choicePrime_train,'test':choicePrime_test})
                
                # isert the raster plot of the topX neurons on each trial type
                if plot_nrns:
                    plot_kwargs = {
                        'pre_time':t*1.1,
                        'post_time':t*.5, 
                        'bin_size':0.01,
                        'smoothing':0.025,
                        'return_fr':True,
                        'baseline_subtract': False}

                    
                    if type(plot_nrns) is str:
                        best_nrn_idx = np.argsort(np.abs(np.mean([choicePrime_train,choicePrime_test],axis=0)))[-5:]
                        nrnIDs = r.cscale[best_nrn_idx]
                    elif type(plot_nrns) is list:
                        nrnIDs = plot_nrns
                        
                    figscale = 5
                    for nrnID in nrnIDs:
                                                    
                        fig,axs = plt.subplots(2,stimGroup.size,figsize=(figscale*stimGroup.size,figscale),sharey='row',sharex=True)

                        for repidx,s in enumerate(stimGroup):
                            ax = axs[0,repidx]
                            ax1 = axs[1,repidx]
                            
                            idx_L = (ev.newIDs==s) & (ev.choiceType==1)
                            idx_R = (ev.newIDs==s) & (ev.choiceType==2)
                            r = my_rasterPSTH(spikes.times,spikes.clusters,[t_on[idx_L],t_on[idx_R]],[nrnID],event_colors=['blue','red'],ax=ax,ax1=ax1,**plot_kwargs)
                        
                            ax.set_title('v:%.0f,a:%.0f,vc:%.0f' % (
                                ev.stim_visAzimuth[idx_L][0],
                                ev.stim_audAzimuth[idx_L][0],                                
                                ev.stim_visContrast[idx_L][0]*100,                                
                            ))

                        namestring = '{subject}_{expDate}_{expNum}_{probeID}'.format(**rec)
                        fig.suptitle('Neuron %.0f, %s, %s' % (nrnID,namestring,onset_time))

                if plot_summary:
                    cmin,cmax = np.nanmin([choicePrime_train,choicePrime_test]),np.nanmax([choicePrime_train,choicePrime_test])
                    #cmin,cmax = cmin*1.1,cmax*1.1
                    c = np.max(np.abs([cmin,cmax]))*1.1
                    #c = 5 # cmin,cmax = -100,100

                    if nOnsets==1:
                        ax=axs1
                    else:
                        ax=axs1[tidx]

                    if 'PeriodOn' in onset_time:
                        tstring = 'stim'
                    else: 
                        tstring = 'movement'

                    ax.plot(choicePrime_train,choicePrime_test,'o',markerfacecolor='cyan',markeredgecolor='k')
                    ax.set_xlim([-c,c])
                    ax.set_ylim([-c,c])
                    ax.axline((0,0),slope=1,color='k',linestyle='--')
                    ax.set_title('R-L,%.1f s befrore %s onset, r=%.2f' % (
                                    t,
                                    tstring,
                                    np.corrcoef(choicePrime_train,choicePrime_test)[0,1]))            
                    ax.set_xlabel('train')
                    ax.set_ylabel('test')
                    namestring = '{subject}_{expDate}_{expNum}_{probeID}'.format(**rec)
                    fig.suptitle(namestring)

        else: 
            dvals={onset_time:None for onset_time in onset_names}    
    except: 
        dvals={onset_time:None for onset_time in onset_names}     

    return Bunch(dvals)