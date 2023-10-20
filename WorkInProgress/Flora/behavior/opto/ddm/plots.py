
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pyddm.functions import solve_partial_conditions

def plot_diagnostics(model=None,sample = None, conditions=None,data_dt =.025,method=None,myloc=0,ax = None,dkwargs = None,mkwargs =None):
    """
    visually assess the diagnostics of the model fit

    """
    if not ax:
        _,ax = plt.subplots(1,1)

    if not dkwargs:
        dkwargs = {
            'alpha' : .5, 
            'color' : 'k'
        }

    if not mkwargs: 
        mkwargs = {
            'lw': 2, 
            'color': 'k'
        }

    if model:
        T_dur = model.T_dur
        if model.dt > data_dt:
            data_dt = model.dt
    elif sample:
        T_dur = max(sample)
    else:
        raise ValueError("Must specify non-empty model or sample in arguments")

    # If a sample is given, plot it behind the model.
    if sample:
        s = sample.subset(**conditions)
        t_domain_data = np.linspace(0, T_dur, int(T_dur/data_dt+1))
        data_hist_top = np.histogram(s.choice_upper, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        data_hist_bot = np.histogram(s.choice_lower, bins=int(T_dur/data_dt)+1, range=(0-data_dt/2, T_dur+data_dt/2))[0]
        total_samples = len(s)
        ax.fill_between(np.asarray(data_hist_top)/total_samples/data_dt+myloc,t_domain_data, label="Data",**dkwargs)
        ax.fill_between(-np.asarray(data_hist_bot)/total_samples/data_dt+myloc,t_domain_data, label="Data", **dkwargs)
        toplabel,bottomlabel = sample.choice_names
    if model:
        s = solve_partial_conditions(model, sample, conditions=conditions, method=method)
        ax.plot(s.pdf("_top")+myloc,model.t_domain(),**mkwargs)
        ax.plot(-s.pdf("_bottom")+myloc,model.t_domain(), **mkwargs)
        toplabel,bottomlabel = model.choice_names


def get_rt_quartiles(m,a,v,o,correct_only = True):
    sol = m.solve(conditions={"audDiff": a, "visDiff": v,'is_laserTrial':o})

    if correct_only:
        correct_side = np.sign(np.sign(a) + np.sign(v)) 
    else:
        correct_side = 0 # when we want to get both

    percentiles = [.48,.5,.52]
    l = [np.interp(np.ptp(sol.cdf('Left'))*p,sol.cdf('Left'),sol.t_domain) for p in percentiles]
    r = [np.interp(np.ptp(sol.cdf('Right'))*p,sol.cdf('Right'),sol.t_domain) for p in percentiles]
    j = [np.interp(np.ptp(sol.cdf('Right')+sol.cdf('Left'))*p,sol.cdf('Right')+sol.cdf('Left'),sol.t_domain) for p in percentiles]

    if correct_side==-1: 
        lower,mid,upper = l
    elif correct_side==1: 
        lower,mid,upper = r
    elif correct_side==0: 
        lower,mid,upper = j

    return lower,mid,upper 

def get_median_rt(sample,a,v,o,correct_only=True):
    dat = sample.subset(audDiff=a,visDiff=v,is_laserTrial=o)
    l = dat.choice_lower
    r = dat.choice_upper 

    if correct_only:
        correct_side = np.sign(np.sign(a) + np.sign(v)) 
    else:
        correct_side = 0 # when we want to get both
    
    if correct_side==-1: 
        rt_ = np.median(l)
    elif correct_side==1: 
        rt_ = np.median(r)
    elif correct_side==0: 
        rt_ = np.median(np.concatenate((l,r)))

    return rt_



def plot_psychometric(model,sample,axctrl=None,axopto=None,plot_log=False): 

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)

    linestyles= ['-','--']
    markerstyles = ['o','x']
    if axctrl is None or axopto is None:         
        _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
    axes = [axctrl,axopto]

    for isLaser,(line,marker,ax) in enumerate(zip(linestyles,markerstyles,axes)):
        psychometric,a,v = zip(*[[model.solve(conditions={"audDiff": a, "visDiff": v,'is_laserTrial':isLaser}).prob('Right'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])
        psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
        psychometric_actual = [sample.subset(audDiff=a,visDiff=v,is_laserTrial=isLaser).prob('Right') for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]
        psychometric_actual = np.reshape(np.array(psychometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 
        psychometric_log = np.log10(psychometric/(1-psychometric))
        #mye = 1e-10
        psychometric_actual_log = np.log10((psychometric_actual+1e-4)/(1-(psychometric_actual)+1e-4))


        colors = ['b','k','r'] # for -1,0,1 aud
        if plot_log:
            [ax.plot(vis_contrasts,p,color=c,linestyle=line) for p,c in zip(psychometric_log,colors)]
            [ax.scatter(actual_vis_contrasts,p,color=c,marker=marker) for p,c in zip(psychometric_actual_log,colors)]
            ax.set_ylim([-3,3])
            ax.axhline(0,color='k',linestyle='--')

        else:
            [ax.plot(vis_contrasts,p,color=c,linestyle=line) for p,c in zip(psychometric,colors)]
            [ax.scatter(actual_vis_contrasts,p,color=c,marker=marker) for p,c in zip(psychometric_actual,colors)]
            ax.set_ylim([-.05,1.05])
            ax.axhline(0.5,color='k',linestyle='--')

        ax.axvline(0,color='k',linestyle='--')
        ax.set_ylabel('p(R)')
        ax.set_xlabel('contrasts')
        ax.set_title('psychometric')


def plot_chronometric(model,sample,axctrl=None,axopto=None):

    actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
    actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
    aud_azimuths  = np.linspace(-1,1,3)
    vis_contrasts = np.linspace(-1,1,40)
    colors = ['b','k','r'] # for -1,0,1 aud

    linestyles= ['-','--']
    markerstyles = ['o','x']
    if axctrl is None or axopto is None:         
        _,(axctrl,axopto) = plt.subplots(2,1,figsize=(20,10))
    axes = [axctrl,axopto]

    for isLaser,(line,marker,ax) in enumerate(zip(linestyles,markerstyles,axes)):

        c_l,c_m,c_u = zip(*[get_rt_quartiles(model,a,v,isLaser,correct_only=False) for a,v in itertools.product(aud_azimuths,vis_contrasts)])
        c_l = np.reshape(np.array(c_l),(aud_azimuths.size,vis_contrasts.size)) 
        c_m = np.reshape(np.array(c_m),(aud_azimuths.size,vis_contrasts.size)) 
        c_u = np.reshape(np.array(c_u),(aud_azimuths.size,vis_contrasts.size)) 

    # or just this way of calculating the chronometric is wrong ohlala, because this is timing on rightward choices, not timing on correct choices
        chronometric_actual = [get_median_rt(sample,a,v,isLaser,correct_only=False) for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]
        chronometric_actual =  np.reshape(np.array(chronometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 





        [ax.fill_between(vis_contrasts, l,u,color=c,alpha=.1,linestyle=line) for l,u,c in zip(c_l,c_u,colors)]
        [ax.scatter(actual_vis_contrasts,chrono,color=c,marker=marker) for chrono,c in zip(chronometric_actual,colors)]

        ax.set_ylabel('mean reaction time')
        ax.set_xlabel('contrasts')
        ax.set_title('chronometric')
    #ax.set_ylim([0.15,.8])



    # fig.suptitle('%s_%s_%s_%s' % (subject,model_name,type,refitted))

    # if to_save:
    #     savename = 'Visualisations/%s_%s_%s_%s.png' % (subject,model_name,type,refitted) 
    #     fig.savefig(basepath / savename,transparent=False,bbox_inches = "tight",format='png',dpi=300)

    # plt.show()
    # print('done')