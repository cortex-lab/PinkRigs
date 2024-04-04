import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_ephys_independent_probes,simplify_recdat,Bunch
from Analysis.pyutils.ev_dat import parse_events
from Analysis.neural.utils.spike_dat import get_binned_rasters



def get_mann_whitneyU(x,y,n_shuffles=20):
    """
    function to calculate the mann-Whitney U(x) statistic for each unit
    Parameters .e.g.
    x = Right choice
    y = left choice
    n_shuffles = n of permitation sets to create

    x: np.ndarray where statistic is compared along the 1st axis (can be as many axes as one wants otherwise e.g. choice x nrn x time)
    y: same as x just for ~choice

    (reason why I input the entire parameter array is because I want to make the permutation sets the same for each nrn)

    """    

    x_y = np.concatenate((x,y),axis=0)
    nx = x.shape[0]

    v_ = np.arange(x_y.shape[0])
    shuffle_idxs = [np.random.permutation(v_)[:nx][np.newaxis,:] for s in range(n_shuffles)]
    shuffle_idxs = np.concatenate(shuffle_idxs)
    shuffle_idxs = np.concatenate((v_[:nx][np.newaxis,:],shuffle_idxs)) # add first row as the actual 
    
    # rank for each nrn
    t = rankdata(x_y,axis=0)
    t= t[shuffle_idxs]

    numer = t.sum(axis=1)
    numer = numer - (nx*(nx+1)/2)

    return numer


def combined_condition_U(spike_counts,trialChoice,trialConditions,n_shuffles=2000):
    """
    function to calculate te combined ranksum across conditions (i.e. cccp anaysis established by Steinmetz et al.)

    Parameters:
    -----------
    spike_counts: np.ndarray : trials x (can be variable dim e.g. trials x nrns or trials x nrns x time etc)
        actual values to rank
    trialChoice: bool np ndarray
        (trials) the two possibilities whose discriminability are comparing in the ROC (e.g. choice under the same stimulus condition)
    trialConditions: np.array 
        (int, trials): unique classes that define which condition the trial belongs to 
    n_shuffles: float (for cv)

    Returns: 
    --------
    : np.ndarray
        U-statistic, with regairds to whatever was considered True by trialChoice
    : np.ndarray
        p-value 
    : np.ndarray
        shuffled U-statisitics


    """
    uCond =np.unique(trialConditions)
    nTotal = np.zeros(((n_shuffles+1,) + spike_counts.shape[1:]))
    dTotal = 0
    for c in uCond: 
        inclT = trialConditions==c
        chA = trialChoice & inclT
        nA = chA.sum()
        chB = ~trialChoice & inclT
        nB = chB.sum()
        uA = get_mann_whitneyU(spike_counts[chA],spike_counts[chB],n_shuffles=n_shuffles)
        nTotal = nTotal+uA
        dTotal = dTotal+nA+nB

    cp = nTotal/dTotal
    t = rankdata(cp,axis=0)
    p = t[0]/(n_shuffles+1)

    return cp[0],p,cp[1:]


class cccp():
    """
    wrapper around ev structures to perform efficient cccp
    """
    def __init__(self):
        self.vis_azimuths = [-60,60]
        self.aud_azimuths =[-60,0,60] 
        self.rt_params = {'rt_min':.05,'rt_max':.35}

    def load_and_format_data(self,rec=None,**kwargs):
        
        if rec is None:
            ephys_dict =  {'spikes': ['times', 'clusters'],'clusters':'all'}
            cameras = ['frontCam','sideCam','eyeCam']
            cam_dict = {cam:{'camera':['times','ROIMotionEnergy']} for cam in cameras}
            cam_dict.update({'events':{'_av_trials':['table']}})
            recordings = load_ephys_independent_probes(ephys_dict=ephys_dict,add_dict=cam_dict,**kwargs)
            rec = recordings.iloc[0]
        
        ev,self.spikes,self.clusters,_,self.cam = simplify_recdat(rec,probe='probe')

        contrasts = np.unique(ev.stim_visContrast)
        contrasts = (contrasts).tolist()

        spls = np.unique(ev.stim_audAmplitude)
        spls = (spls[spls>0]).tolist()


        # parse the events into classes 
        self.ev,self.trial_types = parse_events(
            ev,contrasts=contrasts,
            spls=spls,
            vis_azimuths=self.vis_azimuths,
            aud_azimuths=self.aud_azimuths,
            rt_params=self.rt_params,
            classify_choice_types=True)
        
    
    def get_U(self,test_type='ccCP',t_on_key ='timeline_choiceMoveOn',which_dat='neural',t_before=0.2,t_after=0,t_bin=0.05):
        """
        ccCP = combined condition choice probability
        ccVP = combined condition visual stimulus detction probability
        ccAP = combined condition auditory stimulus detection probability




        Returns: 
        ---------
            :np.ndArray combined U statistic for each unit and time bin the test was conducted in
            :np.ndArray p-value for each unit per time bin
            :np.ndArray shuffled U-statistic for each unit 
            : np.ndArray time bins

            
        """


        trial_types  = self.trial_types
        if 'ccCP' in test_type:
            grouping_indices = trial_types.groupby(by=['contrast','spl','vis_azimuths','aud_azimuths']).indices
        elif 'ccVP' in test_type:
            grouping_indices = trial_types.groupby(by=['contrast','spl','aud_azimuths','choice_type']).indices
        elif 'ccAP' in test_type: 
            df =  trial_types[trial_types.aud_azimuths!=0]  
            grouping_indices = (df).groupby(by=['contrast','spl','vis_azimuths','choice_type']).indices
            grouping_indices = {g:df.index.values[grouping_indices[g]] for g in grouping_indices.keys()}

        # regroup events
        # basically if a group does not contain two indices that means only one type of trial is present
        # so only the groups where there are two grouping indices should really be kept and re_ID-ed
        # and I was thinking we will need a tag but we don't because than e.g. choiceDir will serve as the tag if I did everything right # OMG so cryptic

        ev = self.ev.copy()
        ev.newIDs = np.empty(ev.is_blankTrial.size)*np.nan
        for idx,group in enumerate(grouping_indices.keys()):
            g_idx = grouping_indices[group]
            if len(g_idx)==2:
                # group can be disambiguated
                for g in g_idx:
                    ev.newIDs[ev.trial_type_IDs==g] = idx
        # throw away events that haven't been grouped
        
        ev  = Bunch({k:ev[k][~np.isnan(ev.newIDs)] for k in ev.keys()})

        # trialchoice is what you would shuffle for significance
        if 'ccCP' in test_type:
            trialChoice = ev.timeline_choiceMoveDir==2 # cccp for right choices (Q=is right choice bigger than left choice)
        elif 'ccVP' in test_type:
            trialChoice = ev.stim_visAzimuth==60
        elif 'ccAP' in test_type:
            trialChoice = ev.stim_audAzimuth==60


        if which_dat=='neural':
            raster_kwargs = {
                'pre_time':t_before,
                'post_time':t_after, 
                'bin_size':t_bin,
                'smoothing':0,
                'return_fr':False,
                'baseline_subtract': False, 
                }
            r = get_binned_rasters(self.spikes.times,self.spikes.clusters,self.clusters._av_IDs,ev[t_on_key],**raster_kwargs)
            raster = r.rasters
            tscale = r.tscale

        elif which_dat=='video':
            from Analysis.pyutils.video_dat import get_move_raster
            # get all auditory stimulus onsets
            bin_kwargs  = {
                'pre_time':t_before,
                'post_time':t_after, 
                'bin_size': t_bin,
                'sortAmp':False, 'to_plot':False,
                'baseline_subtract':False
            }


            cam_values = (self.cam.ROIMotionEnergy) # or maybe I should do things based on PCs
            raster,tscale,_  = get_move_raster(ev[t_on_key],self.cam.times,cam_values,**bin_kwargs) 
            


        u,p,u_  = combined_condition_U(raster,trialChoice=trialChoice,trialConditions=ev.newIDs,n_shuffles=100)
        
        return u,p,u_,tscale
    

def get_default_set(which='single_bin'):
    # default set of inputs that are easily editable
        column_names = ['test_type','t_on_key','t_before','t_after','t_bin']
        
        

        if which=='single_bin':
            t_bin_universal = 0.15
            
            params = [
                ('ccAP', 'timeline_audPeriodOn',0,t_bin_universal,t_bin_universal), # should really be taken before rt_params_min
                ('ccCP', 'timeline_choiceMoveOn',t_bin_universal,0,t_bin_universal),
                #('ccCP', 'timeline_audPeriodOn',0.2,0.1,t_bin_universal),
                ('ccVP', 'timeline_audPeriodOn',0,t_bin_universal,t_bin_universal)


            ]
         
        elif which=='multi_bin':
            t_bin_universal = 0.025
            
            params = [
                ('ccAP', 'timeline_audPeriodOn',0,0.2,t_bin_universal), # should really be taken before rt_params_min
                ('ccCP', 'timeline_choiceMoveOn',0.2,0,t_bin_universal),
                #('ccCP', 'timeline_audPeriodOn',0.2,0.1,t_bin_universal),
                ('ccVP', 'timeline_audPeriodOn',0,0.2,t_bin_universal)


            ]
  

        params = pd.DataFrame(params,
            columns= column_names
        )

        return params
