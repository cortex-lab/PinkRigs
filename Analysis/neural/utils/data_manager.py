import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation as mad

from Analysis.neural.utils.video_dat import get_move_raster

# ONE loader from the PinkRig Pipeline
from Admin.csv_queryExp import get_csv_location,load_data


def get_sc_mice_list():
    mice = pd.read_csv(get_csv_location('main'))
    sc_mice = mice[mice.P0_AP<-3.7]
    return sc_mice.Subject.values.tolist()

# a bunch of helper functions for the type pf stuff we want to calculate
def get_performance_metrics(ev):
    """
    Function that calculates performance metrics from a given session 
    

    Parameters: 
    ------------
    ev: Bunch
        the event dict(Bunch) structure read in from the _av_trials.table.parquet
        from the multiSpace session
    
    Returns: 
    ---------
    n_valid_go: numpy ndarray
    n_valid_nogo: numpy ndarray
    coh_performance: numpy ndarray
    vis_performance: numpy ndarray
    aud_performance: numpy ndarray
    tot_bias: numpy ndarray   

    """
    try: 
        n_valid_go = np.sum(ev.is_validTrial[~np.isnan(ev.timeline_choiceMoveDir)])
        n_valid_nogo = np.sum(ev.is_validTrial[np.isnan(ev.timeline_choiceMoveDir)])
        c_idx = (ev.is_validTrial==1) & (ev.is_coherentTrial==1) & ~np.isnan(ev.timeline_choiceMoveDir)
        coh_performance = np.mean((ev.response_feedback[c_idx]+1)/2)
        v_idx = (ev.is_validTrial==1) & (ev.is_visualTrial==1) & ~np.isnan(ev.timeline_choiceMoveDir)
        vis_performance = np.mean((ev.response_feedback[v_idx]+1)/2)
        a_idx = (ev.is_validTrial==1) & (ev.is_auditoryTrial==1) & ~np.isnan(ev.timeline_choiceMoveDir)
        aud_performance = np.mean((ev.response_feedback[a_idx]+1)/2)

        bias_trials = (   (ev.is_validTrial==1) &
                ((ev.is_auditoryTrial==1) | 
                (ev.is_visualTrial==1) | 
                (ev.is_coherentTrial==1))
            )

        left_trials = ((ev.stim_correctResponse[bias_trials])==1)
        right_trials = ((ev.stim_correctResponse[bias_trials])==2)
        trials_to_take = np.min([np.sum(left_trials),np.sum(right_trials)])
        left_trial_choices = ev.timeline_choiceMoveDir[bias_trials][left_trials][:trials_to_take]
        right_trial_choices = ev.timeline_choiceMoveDir[bias_trials][right_trials][:trials_to_take]
        tot_bias = np.nanmean(np.concatenate((left_trial_choices,right_trial_choices)))-1
    except: 
        n_valid_go,n_valid_nogo,coh_performance = None,None,None
        vis_performance,aud_performance,tot_bias = None,None,None

    return (n_valid_go,n_valid_nogo,coh_performance,vis_performance,aud_performance,tot_bias)

def get_rt_metrics(ev):
    try: 
        rt  = ev.timeline_choiceMoveOn - np.nanmin(np.concatenate([ev.timeline_audPeriodOn[:,np.newaxis],ev.timeline_visPeriodOn[:,np.newaxis]],axis=1),axis=1)
        right_rts = rt[ev.timeline_choiceMoveDir==1]
        left_rts = rt[ev.timeline_choiceMoveDir==2]
        r_median,r_mad = np.median(right_rts),mad(right_rts)
        l_median,l_mad = np.median(left_rts),mad(left_rts)
    except: 
        r_median,r_mad = None, None
        l_median,l_mad = None, None

    return (r_median,r_mad,l_median,l_mad)

def get_neural_metrics(clusters): 
    """
    to do: implement is_curated
    """

    if not clusters: 
        n_tot,is_curated,n_good,n_good_metrics = None, None, None, None
    else: 
        try:
            # tot number of units 
            n_tot = clusters.depths.size        
            # check whether it was curated or not 
            is_curated = False
            # check the number of good units after curation/ks
            n_good = (clusters._av_KSLabels==2).sum()
            # get good number of units also by IBL metrics/RP violation and amp gauss cutoff
            n_good_metrics = ((clusters.slidingRP_viol == 1) & (clusters.noise_cutoff>20)).sum()
        except AttributeError:
            n_tot,is_curated,n_good,n_good_metrics = None, None, None, None


    return (n_tot,is_curated,n_good,n_good_metrics)

def get_recorded_channel_position(channels):
    """
    todo: get IBL channels parameter. I think this needs to be implemented on the PinkRig level.
    """
    if not channels: 
        xrange, yrange = None, None
    else:
        xcoords = channels.localCoordinates[:,0]
        ycoords = channels.localCoordinates[:,1]
        xrange = (np.min(xcoords),np.max(xcoords))
        yrange = (np.min(ycoords),np.max(ycoords))

    return (xrange,yrange)

def get_movement_differece(ev,cam): 
    """
    Calculates a measure of facial evoked movement difference between aud L/R trials

    score formula

    z = (r.mean-l.mean/r.std).mean
    score = z[np.argmax(np.abs(z))]

    where r, and l are evoked traces of right and left movements 
    so -ve score: left bias, +ve score: right bias. 

    Parameters: 
    -----------

    Returns:
    --------

    """    
    try: 
        if 'times' not in cam.keys() or 'ROIMotionEnergy' not in cam.keys(): 
            print('there is no camera data')
            score = None 
        elif np.abs(cam.times.size-cam.ROIMotionEnergy.size)>2000:
            print('super different size arrays.')
            score = None
        elif np.unique(cam.ROIMotionEnergy).size==1:
            print('ROI Motion Energy is not changing')
            score=None
        else: 
            rkw = {
                'pre_time':.005,
                'post_time':.2,
                'bin_size':.005, 
                'baseline_subtract': True
                }
            first_vid_length = cam.ROIMotionEnergy.size
            cam.times = cam.times[:first_vid_length]
            on_times = ev.timeline_audPeriodOn[ev.stim_audAzimuth == -60]
            l,_,_ = get_move_raster(on_times, cam.times,cam.ROIMotionEnergy,**rkw)
            on_times = ev.timeline_audPeriodOn[ev.stim_audAzimuth == 60]
            r,_,_ = get_move_raster(on_times, cam.times,cam.ROIMotionEnergy,**rkw)

            zscore_diff = (r.mean(axis=0) -  l.mean(axis=0))/r.std(axis=0)
            score = zscore_diff[np.nanargmax(np.abs(zscore_diff))]
    except: 
        score = None

    return score

def check_postactive(rec):
    """
    For a given active recording, check whether there is a good postactive recording associated with it.  

    Parameters: 
    -----------
    rec: pd.DataFrame 
        csv input from the active recording, containting Subject and expDate
    Returns: 
    ----------
    :str
        expNum of the good passive session.  

    """
    data_dict = {
        'events':{'_av_trials':'table'},
        'eyeCam':{'camera': 'all'}
        }
    any_postactive = load_data(subject = rec.Subject,expDate = rec.expDate,expDef = 'postactive', data_name_dict = data_dict)
    is_long_enough = [float(r.expDuration)>1500 for _,r in any_postactive.iterrows()]
    good_postactive = any_postactive[is_long_enough]
    if len(good_postactive)==1:
        rec = good_postactive.iloc[0]
        postactive_expNum = rec.expNum
        movediff_score = get_movement_differece(rec.events._av_trials,rec.eyeCam.camera)
    else:
        print('postactive could not be unambigously identified...')
        postactive_expNum,movediff_score = None,None 

    return (postactive_expNum,movediff_score)

def determine_ranges_to_merge(ranges):
    current_stop = -1 

    merging = []
    to_merge = []
    for idx,(start, stop) in enumerate(ranges):
        if start > current_stop:
            # this segment starts after the last segment stops
            # just add a new segment
            if len(merging)>0:
                to_merge.append(np.array(merging))
            current_start, current_stop = start, stop
            merging  = [idx]
        else:
            # segments overlap, replace
            # current_start already guaranteed to be lower
            #current_stop = max(current_stop, stop-1)
            merging.append(idx)


    to_merge.append(np.array(merging))

    return to_merge


def check_if_range_overlap(x,y):
    """
    function to check if two sets of ranges overlap 
    Parameters: 
    x,y: tuples of 2: that signify the min and the max of the range
    """

    x = range(int(x[0]),int(x[1]))
    y = range(int(y[0]),int(y[1]))

    is_overlap = not ((x[-1]<y[0]) or (y[-1]<x[0]))

    return is_overlap

def get_highest_yield_unique_ephys(sessions,probe='probe0'):
    """
    function to find the highest yield ephys recordings at each unique position on the probe

    Parameters: 
    ----------- 
    sessions: pd.Dataframe 

    probe: str
        'probe0' or 'probe1'

    Returns:
    --------
        pd.df, list
            two different formats for the recordings that have been selected. 
            Should trim the list version, I think it is used for other functions atm. 

    """
    chosen_rec_list = []
    chosen_rec_df =pd.DataFrame(columns=sessions.columns)
    probe_best = sessions[sessions['%s_n_good' % probe].notna()]
    # sometimes the n_good is  not nan but still the range is
    # throw these away but also warn user

    check_range = probe_best['%s_depth_range' % probe].isna()
    
    if check_range.any(): 
        # warnings
        no_range_recs = probe_best[check_range]
        for _,r in no_range_recs.iterrows(): 
            print('%s %s, expNum=%s does not have channels.localCoordinates but has nrns??!' % (r.Subject,r.expDate,r.expNum))
        # throw
        probe_best = probe_best[probe_best['%s_depth_range' % probe].notna()]

    for s in np.unique(probe_best.Subject):
        subject_table = probe_best[probe_best.Subject == s]
        b = subject_table.pivot(index=['expDate'],columns=['%s_depth_range' % probe],values=['%s_n_good' % probe])
        # merge columns with significant overlap
        ranges = [(int(x[0]),int(x[1])) for x in b['%s_n_good' % probe].columns]
        to_merge = determine_ranges_to_merge(ranges)
            # test if it needs to be merged with anything

        columns_to_keep = ['expDate']
        for columntest in to_merge:
            same_range_columns= b.iloc[:,columntest]
            max_per_column = same_range_columns.max()
            columns_to_keep.append(max_per_column.idxmax()[1])
            
        b = b['%s_n_good' % probe].reset_index()
        b = b[columns_to_keep]
        expdates = b.iloc[:,0]
        for unique_pos_col_idx in range(b.shape[1]-1):
            idx = b.iloc[:,unique_pos_col_idx+1].argmax()   
            chosen_date = expdates[idx]
            selected_recording = subject_table[subject_table.expDate==chosen_date]            
            chosen_rec_list.append((s,chosen_date,selected_recording.expNum.values[0],probe)) # this does not work anymore as I am trying to use this for the active as well
            chosen_rec_df = pd.concat((chosen_rec_df,selected_recording))

    chosen_rec_df = chosen_rec_df.reset_index()

    return chosen_rec_df,chosen_rec_list

def get_behavior_quality_scores(savepath=None,trim_bad=False,n_go_thr = 200,performance_cutoff = .65,bias_cutoff = .6,good_nrn_thr = 10,rt_diff_thr = 0.1,**kwargs):

    """
    create csv of all recordings of interst 
    calculate metrics of performance and ephys quality 
    basically creating csv for batch analysis 

    Parameters: 
    ----------
    savepath: pathlib.Path
        path to save the csv to
    trim_bad: bool 
        whether to throw away recordings based on below criteria
    n_go_thr: float
        min. number of go trials 
    performance_cutoff: float,0-1
        min. performance applied separately to aud,vis & coherent trials 
    bias_cutoff: float, 0.5-1
        >.5 bias means right bias --> this number trims both the right and left bias equally
    good_nrn_thr: float
        min. # of good neurons. atm this is the kilosort good neurons
    rt_diff_thr: float, in sec
        max. reaction time time difference between the two sides
    
    Returns: 
    ----------
    pd.df
        recording behaviour and ephys quality metrics 

    """    
    # after implant measures only
    data_dict = {
        'events':{'_av_trials':'table'},
        'probe0':{'clusters':'all'},
        'probe1':{'clusters':'all'},
        'eyeCam':{'camera': 'all'},
        'probe0_raw':{'channels':'all'},
        'probe1_raw':{'channels':'all'}
        }

    kwargs['expDef'] = 'multiSpace'
    recdat = load_data(data_name_dict = data_dict,**kwargs)
    # from recdat drop recordings that are too short
    out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration']]    
    out_dat = out_dat.reset_index(drop=True)
    # performance measures   
    go,nogo,p_coh,p_vis,p_aud,bias = zip(*[get_performance_metrics(rec.events._av_trials) for _,rec in recdat.iterrows()])
    # reaction time measures 
    rt_r,rt_r_mad,rt_l,rt_l_mad = zip(*[get_rt_metrics(rec.events._av_trials) for _,rec in recdat.iterrows()])
    # recording locations 
    probe0_shank_range,probe0_depth_range = zip(*[get_recorded_channel_position(rec.probe0_raw.channels) for _,rec in recdat.iterrows()])
    probe1_shank_range,probe1_depth_range = zip(*[get_recorded_channel_position(rec.probe1_raw.channels) for _,rec in recdat.iterrows()])
    # neural metric measures
    n_tot0,is_curated0,n_good0,n_good_metrics0 = zip(*[get_neural_metrics(rec.probe0.clusters) for _,rec in recdat.iterrows()])
    n_tot1,is_curated1,n_good1,n_good_metrics1 = zip(*[get_neural_metrics(rec.probe1.clusters) for _,rec in recdat.iterrows()])
    # camera movement meausres
    cam_score_active = [get_movement_differece(rec.events._av_trials,rec.eyeCam.camera) for _,rec in recdat.iterrows()] 
    # metrics on passive 
    passive_expNums,cam_score_passive = zip(*[check_postactive(rec) for _,rec in recdat.iterrows()])
    
    # add all to dataframe as columns
    out_dat = out_dat.assign(
        n_go = go,
        n_nogo = nogo,
        performance_aud = p_aud, 
        performance_vis = p_vis,
        performance_coherent = p_coh,
        choice_bias = bias,
        rt_right_choice = rt_r, 
        rt_left_choice = rt_l, 
        rt_mad_right_choice =rt_r_mad,
        rt_mad_left_choice = rt_l_mad, 
        probe0_shank_range = probe0_shank_range, 
        probe0_depth_range = probe0_depth_range,
        probe1_shank_range = probe1_shank_range, 
        probe1_depth_range = probe1_depth_range,
        probe0_n_total = n_tot0, 
        probe0_is_curated = is_curated0, 
        probe0_n_good = n_good0,
        probe0_n_good_metrics = n_good_metrics0,
        probe1_n_total = n_tot1, 
        probe1_is_curated = is_curated1, 
        probe1_n_good = n_good1,
        probe1_n_good_metrics = n_good_metrics1,
        passive_expNum = passive_expNums, 
        movediff_score_active = cam_score_active, 
        movediff_score_passive = cam_score_passive
        )

    if savepath: 
        out_dat.to_csv(savepath)

    if trim_bad: 
        is_good_rec = (
            (out_dat.n_go>n_go_thr) &
            (out_dat.performance_aud>performance_cutoff) & 
            (out_dat.performance_vis>performance_cutoff) & 
            (out_dat.performance_coherent>performance_cutoff) & 
            (out_dat.choice_bias<bias_cutoff) & # right bias 
            (out_dat.choice_bias>(1-bias_cutoff)) & # left bias
            (np.abs(out_dat.rt_mad_left_choice-out_dat.rt_mad_right_choice)<rt_diff_thr) & 
            (out_dat.probe0_n_total.fillna(0)+out_dat.probe1_n_total.fillna(0) >good_nrn_thr) 
            )
            
        out_dat = out_dat[is_good_rec]


    return out_dat


def get_sessions_with_units(expdef_namestring,savepath=None,trim_bad = False,**kwargs): 
    """
    this function aims to  get the valid postactive experiments postImplant
    with the aim that you are then able to sort according to crietria

    Parameters: 
    ----------
    exdefnamestring: str
        name of expdef

    Returns: 
        :pd.DataFrame
    
    """

    data_dict = {
    'probe0':{'clusters':'all'},
    'probe1':{'clusters':'all'},
    'probe0_raw':{'channels':'localCoordinates'},
    'probe1_raw':{'channels':'localCoordinates'}
    }

    kwargs['expDef'] = expdef_namestring
    recdat = load_data(data_name_dict = data_dict,**kwargs)
    # from recdat drop recordings that are too short
    out_dat = recdat[['Subject','expDate','expNum','rigName','expDuration']]    
    out_dat = out_dat.reset_index(drop=True)

    # recording locations 
    probe0_shank_range, probe0_depth_range  = zip(*[get_recorded_channel_position(rec.probe0_raw.channels) for _,rec in recdat.iterrows()])
    probe1_shank_range, probe1_depth_range  = zip(*[get_recorded_channel_position(rec.probe1_raw.channels) for _,rec in recdat.iterrows()])
    # neural metric measures
    n_tot0,is_curated0,n_good0,n_good_metrics0 = zip(*[get_neural_metrics(rec.probe0.clusters) for _,rec in recdat.iterrows()])
    n_tot1,is_curated1,n_good1,n_good_metrics1 = zip(*[get_neural_metrics(rec.probe1.clusters) for _,rec in recdat.iterrows()])

        # add all to dataframe as columns
    out_dat = out_dat.assign(
        probe0_shank_range = probe0_shank_range, 
        probe0_depth_range = probe0_depth_range,
        probe1_shank_range = probe1_shank_range, 
        probe1_depth_range = probe1_depth_range,
        probe0_n_total = n_tot0, 
        probe0_is_curated = is_curated0, 
        probe0_n_good = n_good0,
        probe0_n_good_metrics = n_good_metrics0,
        probe1_n_total = n_tot1, 
        probe1_is_curated = is_curated1, 
        probe1_n_good = n_good1,
        probe1_n_good_metrics = n_good_metrics1
        )

    if savepath: 
        out_dat.to_csv(savepath)

    if trim_bad:
        print('have to implement some arbitrary threshold measures for selectinf data for analysis.')

    print('almost done.')



    return out_dat

def simplify_recdat(recording,probe_dat_type='probe0'): 
    """
    spits out the event,spike etc bunches with one line
    """
    


    ev,spikes,clusters,channels = None,None,None,None
    if hasattr(recording,'events'):
        ev = recording.events._av_trials

    if hasattr(recording,probe_dat_type):
        p_dat = recording[probe_dat_type]
        if hasattr(p_dat,'spikes'):
            spikes = p_dat.spikes
        
        if hasattr(p_dat,'clusters'):
            clusters = p_dat.clusters
        
        if hasattr(p_dat,'channels'):
            channels = p_dat.channels

    return (ev,spikes,clusters,channels)


def load_cluster_info(probe = 'probe0',**rec_kwargs): 
    """
    function to collect *all* the cluster info we hold into one single dataFrame 
    (including anat files from raw location etc.)

    parameters:   

    returns: 
        : pd.DataFrame
        
    """
    data_dict = {
        probe:{'clusters':'all'}, 
        (probe + '_raw'):{'clusters':['brainLocationAcronyms_ccf_2017', 'brainLocationIds_ccf_2017','mlapdv' ] }
    }
    recording = load_data(data_name_dict=data_dict,**rec_kwargs)
    clusters = recording[probe].iloc[0].clusters
    clusters_r = recording[(probe + '_raw')].iloc[0].clusters

    clusInfo = {k:clusters[k] for k in clusters.keys() if clusters[k].ndim==1}
    clusInfo = pd.DataFrame.from_dict(clusInfo)
    clusInfo = clusInfo.set_index('_av_IDs',drop=False)


    if 'mlapdv' in list(clusters_r.keys()):
        clusInfo_ = {k:clusters_r[k] for k in clusters_r.keys() if clusters_r[k].ndim==1}
        clusInfo_ = pd.DataFrame.from_dict(clusInfo_)
        clusInfo_['ml'] = clusters_r.mlapdv[:,0]
        clusInfo_['ap'] = clusters_r.mlapdv[:,1]
        clusInfo_['dv'] = clusters_r.mlapdv[:,2]
        all_clusInfo = pd.concat([clusInfo,clusInfo_],axis=1)
        all_clusInfo = all_clusInfo.loc[clusInfo.index]
    else: 
        all_clusInfo = clusInfo    


    # also try loading in the shank posititon this is for the naive
    probe_imec = 'imec0' if 'probe0' in probe else 'imec1'

    sc_probeloc_path = Path(r'C:\Users\Flora\Documents\Processed data\passiveAV_project')
    registration_folder = sc_probeloc_path / rec_kwargs['subject'] / rec_kwargs['expDate']/ 'alf' / probe_imec
    registration_files = list(registration_folder.glob('*.npy')) 

    if len(registration_files)==4:  
        print('acute recording. Found SC registration.') 
        d = {}
        for i,r in enumerate(registration_files):
            d[i] = np.load(r)

        # and now assign each for the unit. 
        all_clusInfo['sc_azimuth'] = [d[s][0] for s in all_clusInfo._av_shankID]
        all_clusInfo['sc_elevation'] = [d[s][1] for s in all_clusInfo._av_shankID]
        all_clusInfo['sc_surface'] = [d[s][2] for s in all_clusInfo._av_shankID]

    elif len(registration_files)==0:
        print('trying to load a chronic registration ...')

        sc_probeloc_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
        registration_folder = sc_probeloc_path / rec_kwargs['subject'] 
        registration_files = list(registration_folder.glob('%s_*.npy' % probe_imec))

        if len(registration_files)==1:
            long_form_dat = np.load(registration_files[0])

            d = {}
            for i,r in enumerate(long_form_dat):
                d[i] = r
            # things are in a different order!!!!!!
            all_clusInfo['sc_azimuth'] = [d[s][1] for s in all_clusInfo._av_shankID]
            all_clusInfo['sc_elevation'] = [d[s][2] for s in all_clusInfo._av_shankID]
            all_clusInfo['sc_surface'] = [d[s][0] for s in all_clusInfo._av_shankID]
    

    return all_clusInfo


       



