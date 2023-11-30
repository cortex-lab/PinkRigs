import numpy as np
import xarray as xr 
import itertools
import pandas as pd 


from Analysis.neural.utils.spike_dat import bincount2D
from Admin.csv_queryExp import Bunch


def postactive(ev):  

        active_session =  hasattr(ev,'timeline_choiceMoveDir')  

        blank_times = ev.block_stimOn[ev.is_blankTrial==1]

        timepoints_blanks=xr.DataArray(blank_times[:,np.newaxis],
                                dims=('trials','timeID'),
                                coords={'timeID':['ontimes']})



        if active_session: 
                trial_no =  40       
        else: 
                trial_no = blank_times.size
        
        azimuths_set = np.unique(ev.stim_visAzimuth)
        azimuths_set = azimuths_set[[not np.isnan(elem) for elem in azimuths_set]]
        contrast_set = np.unique(ev.stim_visContrast)
        contrast_set = contrast_set[contrast_set>0]
        spl_set = np.unique(ev.stim_audAmplitude)
        spl_set = spl_set[spl_set>0]

        # visual
        myarray=np.zeros((azimuths_set.size,contrast_set.size,trial_no,1))

        for i,myazi in enumerate(azimuths_set):
                for j,myContrast in enumerate(contrast_set):
                        ix=np.where((ev.is_visualTrial==1) & 
                                (ev.stim_visAzimuth==myazi) &
                                (ev.stim_visContrast==myContrast)
                                )[0] 
                        timepoints = ev.timeline_visPeriodOn[ix]
                        if timepoints.size!=trial_no: 
                            timepoints = np.append(timepoints,np.empty((trial_no-timepoints.size))*np.nan)
   
                        try: 
                                myarray[i,j,:,0]=timepoints
                        except ValueError: 
                                print('%.2f deg azimuth, %.2f Contrast combination does not exist' % (myazi,myContrast))



                timepoints_visual=xr.DataArray(myarray,
                                dims=('azimuths','contrast','trials','timeID'),
                                coords={'azimuths':azimuths_set,
                                        'contrast':contrast_set,
                                        'timeID':['ontimes']})

        # auditory
        myarray=np.zeros((azimuths_set.size,spl_set.size,trial_no,1))
        for i,myazi in enumerate(azimuths_set):
                for j,mySPL in enumerate(spl_set):
                        ix=np.where((ev.is_auditoryTrial==1) & 
                                (ev.stim_audAzimuth==myazi) &
                                (ev.stim_audAmplitude==mySPL)
                                )[0] 
                        timepoints =ev.timeline_audPeriodOn[ix]
                        if timepoints.size!=trial_no: 
                            timepoints = np.append(timepoints,np.empty((trial_no-timepoints.size))*np.nan)
                        myarray[i,j,:,0] = timepoints

                timepoints_audio=xr.DataArray(myarray,
                                dims=('azimuths','SPL','trials','timeID'),
                                coords={'azimuths':azimuths_set,
                                        'SPL':spl_set,
                                        'timeID':['ontimes']})


        #audiovisual
        myarray=np.zeros((azimuths_set.size,azimuths_set.size,contrast_set.size,spl_set.size,trial_no,2))
        for i,myaziVIS in enumerate(azimuths_set):
                for j, myaziAUD in enumerate(azimuths_set): 
                        for k,myContrast in enumerate(contrast_set): 
                                for l,mySPL in enumerate(spl_set):
                                        ix=np.where(((ev.is_coherentTrial==1) | 
                                                        (ev.is_conflictTrial==1)) & 
                                                        (ev.stim_visAzimuth ==myaziVIS) &
                                                        (ev.stim_audAzimuth==myaziAUD) &
                                                        (ev.stim_audAmplitude==mySPL) &
                                                        (ev.stim_visContrast == myContrast)
                                                        )[0]

                                        if ix.size>0: 
                                                timepoints_V = ev.timeline_visPeriodOn[ix[:trial_no]]
                                                if timepoints_V.size!=trial_no: 
                                                        timepoints_V = np.append(timepoints_V,np.empty((trial_no-timepoints_V.size))*np.nan)

                                                timepoints_A = ev.timeline_audPeriodOn[ix[:trial_no]]
                                                if timepoints_A.size!=trial_no: 
                                                        timepoints_A = np.append(timepoints_A,np.empty((trial_no-timepoints_A.size))*np.nan)

                                                myarray[i,j,k,l,:,0]=timepoints_V
                                                myarray[i,j,k,l,:,1]= timepoints_A

                
        myarray[myarray==0]=np.nan

        MSvisAzimuth = ev.stim_visAzimuth[((ev.is_coherentTrial==1) | (ev.is_conflictTrial==1))]
        MSaudAzimuth = ev.stim_audAzimuth[((ev.is_coherentTrial==1) | (ev.is_conflictTrial==1))]
        uniquecomb=np.unique(np.array([MSvisAzimuth,MSaudAzimuth]),axis=1)
        congruent_ix=np.where(uniquecomb[0,:]==uniquecomb[1,:]) 
        incongruent_ix=np.where(uniquecomb[0,:]!=uniquecomb[1,:])

        timepoints_MS=xr.DataArray(myarray,
                                dims=('visazimuths','audazimuths','contrast','SPL','trials','timeID'),
                                coords={'visazimuths':azimuths_set,
                                        'audazimuths':azimuths_set,
                                        'contrast':contrast_set,
                                        'SPL':spl_set,
                                        'timeID':['visontimes','audontimes']},
                                attrs={'congruent_azimuths': uniquecomb[0,congruent_ix], 
                                        'incongruent_vis_azimuths': uniquecomb[0,incongruent_ix],
                                        'incongruent_aud_azimuths': uniquecomb[1,incongruent_ix]})






        return timepoints_blanks,timepoints_visual,timepoints_audio,timepoints_MS

def index_trialtype_perazimuth(a,v,expType='active'):
        """
        function to index into dictionnary of trial types sorted by arrange_trials based on azimuth
        a: auditory azimuth
        v: visual azimuth

        returns the type of trial: blank/vis/aud/coh/conf, string    
        
        key reason to distinguish active and passive is behcause on active visual trials are a=0 v!=1000 
        vs on passive a=-1000 and v!=-1000
        # probably this is somewhat redundant anyway....
        """   


        if expType=='active':  
                if (a==0) & (v==-1000):
                        trialtype = 'is_blankTrial'
                elif (a==0) & (v!=-1000):
                        trialtype = 'is_visualTrial'
                elif (a!=0) & (v==-1000):
                        trialtype = 'is_auditoryTrial'
                elif (a!=0) & (v!=-1000) & (v==a):
                        trialtype = 'is_coherentTrial'
                elif (a!=0) & (v!=-1000) & (v!=a):
                        trialtype = 'is_conflictTrial'

        elif expType=='passive': 
                # basically we just replace nans with uncrealistic numbers.
                if np.isnan(a):
                        a=-1000
                if np.isnan(v):
                        v=-1000

                if (a==-1000) & (v==-1000):
                        trialtype = 'is_blankTrial'
                elif (a==-1000) & (v!=-1000):
                        trialtype = 'is_visualTrial'
                elif (a!=-1000) & (v==-1000):
                        trialtype = 'is_auditoryTrial'
                elif (a!=-1000) & (v!=-1000) & (v==a):
                        trialtype = 'is_coherentTrial'
                elif (a!=-1000) & (v!=-1000) & (v!=a):
                        trialtype = 'is_conflictTrial'

        return trialtype

def digitise_event_onsets(ev_times,bin_range = None,**binkwargs): 
    """
    function to digitise event onsets and if called return bin indices.
    
    """
    myev_digi,t,_ = bincount2D(ev_times,np.ones(ev_times.size),**binkwargs)       
    onset_idx = np.where(myev_digi[0,:]==1)[0]

    if bin_range: 
        t_before, t_after = bin_range[0], bin_range[1]
        t_bin = binkwargs['xbin']
        bin_range = np.arange(t_before/t_bin,t_after/t_bin).astype('int')
        bin2show  = [(bin_range + idx)[np.newaxis,:] for idx in onset_idx]
        bin2show = np.concatenate(bin2show,axis=0)
    else: 
        bin2show = None

    return myev_digi,onset_idx,bin2show

def digitize_events(ontimes,offtimes,timepoints):
    """
    another function to digitise events between on and offsets
    
    """
    ontimes = ontimes[~np.isnan(ontimes)]
    offtimes = offtimes[~np.isnan(offtimes)]                  
    new_dt = np.diff(timepoints).mean()
    FIRST_EVENT= timepoints.min()  # because the first event of the neural data is not always 0 

    events_nsamples = timepoints.size
    onsets,offsets= ontimes-FIRST_EVENT, offtimes-FIRST_EVENT
        
    events_digitized = np.zeros(events_nsamples, dtype=np.uint8)
    for idx, (on, off) in enumerate(zip(onsets, offsets)):
        onsample = int(on/new_dt)
        offsample = int((off)/new_dt) # (off+new_dt) is there to include the last digital sample
        if idx % 10000 == 0:
            # print every 1000th event
            print(idx, ontimes[idx], offtimes[idx], timepoints[onsample], timepoints[offsample])
        events_digitized[onsample:offsample] = 1
    

    return events_digitized

def getTrialNames(ev):
        """
        in the ev structure we have trialtypes in a binary format but sometimes this is more useful to have as a list for pd/seaborn 
        so this function returns the list

        """
        trialtype = np.sum(np.concatenate((
        (ev.is_blankTrial*1)[np.newaxis,:],
        (ev.is_auditoryTrial*2)[np.newaxis,:], 
        (ev.is_visualTrial*3)[np.newaxis,:],
        (ev.is_coherentTrial*4)[np.newaxis,:],
        (ev.is_conflictTrial*5)[np.newaxis,:]
        )),axis=0)

        #
        trialNames = [] 
        for i in trialtype: 
                if i==0: 
                        n='noStim'
                elif i==1:
                        n='blank'
                elif i==2: 
                        n='auditory'
                elif i==3:
                        n='visual'
                elif i==4:
                        n='coherent'
                elif i==5:
                        n='conflict'

                trialNames.append(n)   

        return trialNames 





def parse_events(ev,contrasts,spls,vis_azimuths,aud_azimuths,
                classify_choice_types=True,choice_types = None, 
                rt_params = None, 
                classify_rt = False, 
                min_trial = 2, 
                include_unisensory_aud = True, 
                include_unisensory_vis = False,add_crossval_idx_per_class=False):
        """
        function that preselects some events based on parameters
          1) excludes events that we don't intend to fit at all
          2) identifies events into sub-categories such that 
            a.) during balanced cross-validation we are able to allocate trialtypes into both train & test sets
            b.) we are able to equalise how many of each of these trial type go into the model at all -- it is e.g. unfair to fill the model with a lot of correct choices and few incorrect choices when wanting to fit 'choice'     

            
        Default exclusions: 
                -  invalid trials
                -  when mouse has a change of mind
        Optional exclusions: 
                - RT is not within range defined by rt_params

        Parameters: 
        classify_choice_types: bool
                - whether to split trials by choice type
        choice_type: list
                which choices to split to 
        rt_params: None/dict
                if dict, it must contain keys 'rt_min' and 'rt_max'
                defines ranges of rt bsed on which a trial is included or not
        min_trial: int
            if we rebalance across trial types this is the minimum trial no. we require from each trial type.   
        include_unisensory_aud: bool
        include_unisensory_vis: bool
        add_crossval_idx_per_class: bool
            whether to split each trial type individially to training and test set

        """

        # keep trials or not based on global criteria 

        ev_ = ev.copy()

        to_keep_trials = ev.is_validTrial.astype('bool')

        # if it is active data, also exclude trials where firstmove was made prior to choiceMove
        if hasattr(ev,'timeline_firstMoveOn'):
            no_premature_wheel = (ev.timeline_firstMoveOn-ev.timeline_choiceMoveOn)==0
            no_premature_wheel = no_premature_wheel + np.isnan(ev.timeline_choiceMoveOn) # also add the nogos
            to_keep_trials = to_keep_trials & no_premature_wheel

        
        if rt_params:
                if rt_params['rt_min']: 
                        to_keep_trials = to_keep_trials & (ev.rt>=rt_params['rt_min'])
                
                if rt_params['rt_max']: 
                        to_keep_trials = to_keep_trials & (ev.rt<=rt_params['rt_max'])   

         # and if there is anything else wrong with the trial, like the pd did not get detected..? 

         # and if loose rebalancing strategy is called for choices i.e. something that just factors bias away 

        ev  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})


        ##### TRIAL SORTING #########
        # create the global criteria 
        if classify_choice_types & hasattr(ev,'timeline_choiceMoveDir'):
            # active
            ev.choiceType = ev.timeline_choiceMoveDir
            ev.choiceType[np.isnan(ev.choiceType)] = 0
            if not choice_types:
                choice_types = np.unique(ev.choiceType) 

        else:
            # passive (animal did not go)
            ev.choiceType = np.zeros(ev.is_validTrial.size).astype('int')
            choice_types = [0]
        

        if classify_rt:
            # option to balance whether trial types are dominantly long/short in a certain trial class
            pass

        if include_unisensory_aud:                 
            # in unisensory cases the spl/contrast is set to 0 as expected, however the azimuths are also set to nan -- not the easiest to query...
            ev.stim_visAzimuth[np.isnan(ev.stim_visAzimuth)] =-1000 

            vis_azimuths.append(-1000)
            vis_azimuths.sort()
            
            spls.append(0)
            spls.sort()             
        
        if include_unisensory_vis:
            ev.stim_audAzimuth[np.isnan(ev.stim_audAzimuth)] =-1000

            aud_azimuths.append(-1000)
            aud_azimuths.sort()

            contrasts.append(0)
            contrasts.sort()

        # separate trials into trial classes
        trial_classes = {}
        for idx,(c,spl,v_azi,a_azi,choice) in enumerate(itertools.product(contrasts,spls,vis_azimuths,aud_azimuths,choice_types)):
            # create a dict

            is_this_trial = ((ev.stim_visContrast == c) &
                            (ev.stim_audAmplitude == spl) &
                            (ev.stim_visAzimuth == v_azi) & 
                            (ev.stim_audAzimuth == a_azi) & 
                            (ev.choiceType == choice))
            

            trial_classes[idx] =Bunch ({'is_this_trial':is_this_trial,
                                        'contrast':c,
                                        'spl':spl,
                                        'vis_azimuths':v_azi,
                                        'aud_azimuth':a_azi,
                                        'choice_type': choice,
                                        'n_trials':is_this_trial.sum()})
            
        
            
        # check how balanced the data is....
        #print('attempting to rebalance trials ...')
        trial_class_IDs  = np.array(list(trial_classes.keys()))
        n_in_class = np.array([trial_classes[idx].n_trials for idx in trial_class_IDs])

        # some requested classes don't actually need to be fitted ...
        #print('%.0d/%0d requested trial types have 0 trials in it...' % ((n_in_class==0).sum(),len(trial_classes)))


        min_test  = ((n_in_class>0) & (n_in_class<min_trial))

        if min_test.sum()>0:
            print('some types do not pass the minimum trial requirement. Lower min requirement or pass more data.')
            #trial_class_IDs[min_test]

        # allocate each trialType to train/test set   
        kept_trial_class_IDs = trial_class_IDs[(n_in_class>=min_trial)]
        trial_classes = Bunch({k:trial_classes[k] for k in kept_trial_class_IDs})

        if add_crossval_idx_per_class:
        # divide each trial type into cv sets
                for k in kept_trial_class_IDs: 
                        curr_trial_idx = np.where(trial_classes[k].is_this_trial)[0]

                        np.random.seed(0)
                        np.random.shuffle(curr_trial_idx)
                        middle_index = curr_trial_idx.size//2 
                        train_idx = curr_trial_idx[:middle_index]
                        test_idx = curr_trial_idx[middle_index:] 
                        cv_inds = np.empty(trial_classes[k].is_this_trial.size) * np.nan

                        cv_inds[train_idx] = 1 
                        cv_inds[test_idx] =  2

                        #train=(cv_inds[train_idx,:]*1).sum(axis=0).astype('int')
                        #test =(cv_inds[test_idx,:]*2).sum(axis=0).astype('int')
                        trial_classes[k]['cv_inds'] = cv_inds
       
        # pass on events to the kernels        

        trial_classes = Bunch(trial_classes)
                       
        to_keep_trials = np.sum(np.array([trial_classes[idx].is_this_trial for idx in kept_trial_class_IDs]),axis=0).astype('bool')
        
        # add the trial index type
        ev.trial_type_IDs = np.sum(np.concatenate([((trial_classes[idx].is_this_trial)*(ct))[np.newaxis,:] for ct,idx in enumerate(kept_trial_class_IDs)]),axis=0)

        if add_crossval_idx_per_class: 
                ev.cv_set = np.nansum(np.array([trial_classes[idx].cv_inds for idx in kept_trial_class_IDs]),axis=0) # no. of trials allocated to each class can be still slightly uneven if the trial numbers are odd

        ev  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})

        #print('%.0d trials/%.0d trials are kept.' % (ev.is_validTrial.sum(),ev_.is_validTrial.sum()))

        # recalculate kernel_trialOn and kernel_trialOff, taking into account some all the things that possibly go into the kernel
        # get the df to annotate the trial classes in long form
        contrasts_,spls_,vis_azimuths_,aud_azimuths_,choice_types_,n_trials_=  zip(*[
        (trial_classes[idx].contrast,
                trial_classes[idx].spl,
                trial_classes[idx].vis_azimuths,
                trial_classes[idx].aud_azimuth,
                trial_classes[idx].choice_type,
                trial_classes[idx].n_trials,
                ) 
        for idx in kept_trial_class_IDs])
        
        class_types = pd.DataFrame({
        'contrast': contrasts_, 
        'spl':spls_, 
        'vis_azimuths':vis_azimuths_,
        'aud_azimuths':aud_azimuths_,
        'choice_type':choice_types_,
        'n_trials_':n_trials_
        })



        return ev,class_types
