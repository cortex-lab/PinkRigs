#%% 

# read in the result of the plot & plot the pyschometric, the chonoetric and other diagnostics
import numpy as np
from pathlib import Path
import pickle 
import itertools
import matplotlib.pyplot as plt

model_name = 'DriftAdditiveSplit' 

#subjects = ['AV036','AV038','AV041','AV046','AV047']
subject = 'AV038'

#for subject in subjects:
type = 'Opto'
refitted = 'Ctrl'
plot_log = True
to_save = True

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto')
savepath = basepath / model_name

if 'Opto' in type:
    if 'Ctrl' in refitted:
        model_path = (savepath / ('%s_CtrlModel.pickle' % (subject)))
    else:
        model_path = (savepath / ('%s_OptoModel_%s.pickle' % (subject,refitted)))
    sample_path = (savepath / ('%s_%sSample_train.pickle' % (subject,type)))

elif 'Ctrl' in type:
    model_path = (savepath / ('%s_CtrlModel.pickle' % (subject)))
    sample_path = (savepath / ('%s_%sSample.pickle' % (subject,type)))

with open(model_path.__str__(),'rb') as f:
    m = pickle.load(f)
with open(sample_path.__str__(),'rb') as f:
    sample = pickle.load(f)

actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))

#
from WorkInProgress.Flora.behavior.opto.ddm.fitting import get_params_fixation
get_params_fixation(m)

print('fit result,%s : %.2f' % (m.fitresult.loss,m.fitresult.value()))
#pyddm.plot.model_gui(model=m, sample=sample, conditions={"audDiff": actual_aud_azimuths, "visDiff": actual_vis_contrasts})

# %%
# inspect data vs predicition

aud_azimuths  = np.linspace(-1,1,3)
vis_contrasts = np.linspace(-1,1,40)

psychometric,a,v = zip(*[[m.solve(conditions={"audDiff": a, "visDiff": v}).prob('Right'),a,v] for a,v in itertools.product(aud_azimuths,vis_contrasts)])
psychometric = np.reshape(np.array(psychometric),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
a = np.reshape(np.array(a),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
v = np.reshape(np.array(v),(aud_azimuths.size,vis_contrasts.size)) # reshape to aud x vis matrix 
psychometric_actual = [sample.subset(audDiff=a,visDiff=v).prob('Right') for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]
psychometric_actual = np.reshape(np.array(psychometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 
psychometric_log = np.log10(psychometric/(1-psychometric))
#mye = 1e-10
psychometric_actual_log = np.log10((psychometric_actual+1e-4)/(1-(psychometric_actual)+1e-4))



def get_rt_quartiles(m,a,v,correct_only = True):
    sol = m.solve(conditions={"audDiff": a, "visDiff": v})

    if correct_only:
        correct_side = np.sign(np.sign(a) + np.sign(v)) 
    else:
        correct_side = 0 # when we want to get both

    percentiles = [.25,.5,.75]
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

def get_median_rt(sample,a,v,correct_only=True):
    dat = sample.subset(audDiff=a,visDiff=v)
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

c_l,c_m,c_u = zip(*[get_rt_quartiles(m,a,v,correct_only=False) for a,v in itertools.product(aud_azimuths,vis_contrasts)])
c_l = np.reshape(np.array(c_l),(aud_azimuths.size,vis_contrasts.size)) 
c_m = np.reshape(np.array(c_m),(aud_azimuths.size,vis_contrasts.size)) 
c_u = np.reshape(np.array(c_u),(aud_azimuths.size,vis_contrasts.size)) 



# or just this way of calculating the chronometric is wrong ohlala, because this is timing on rightward choices, not timing on correct choices


chronometric_actual = [get_median_rt(sample,a,v,correct_only=False) for a,v in itertools.product(actual_aud_azimuths,actual_vis_contrasts)]

chronometric_actual =  np.reshape(np.array(chronometric_actual),(actual_aud_azimuths.size,actual_vis_contrasts.size)) 



fig,(ax,axc) = plt.subplots(1,2,figsize=(20,10))
colors = ['b','k','r'] # for -1,0,1 aud
if plot_log:
    [ax.plot(vis_contrasts,p,color=c) for p,c in zip(psychometric_log,colors)]
    [ax.scatter(actual_vis_contrasts,p,color=c,marker='o') for p,c in zip(psychometric_actual_log,colors)]
    ax.set_ylim([-3,3])
    ax.axhline(0,color='k',linestyle='--')

else:
    [ax.plot(vis_contrasts,p,color=c) for p,c in zip(psychometric,colors)]
    [ax.scatter(actual_vis_contrasts,p,color=c,marker='o') for p,c in zip(psychometric_actual,colors)]
    ax.set_ylim([-.05,1.05])
    ax.axhline(0.5,color='k',linestyle='--')




ax.axvline(0,color='k',linestyle='--')
ax.set_ylabel('p(R)')
ax.set_xlabel('contrasts')
ax.set_title('psychometric')
#[axc.plot(vis_contrasts,chrono,color=c) for chrono,c in zip(chronometric,colors)]


[axc.fill_between(vis_contrasts, l,u,color=c,alpha=.1) for l,u,c in zip(c_l,c_u,colors)]
[axc.scatter(actual_vis_contrasts,chrono,color=c,marker='o') for chrono,c in zip(chronometric_actual,colors)]

axc.set_ylabel('mean reaction time')
axc.set_xlabel('contrasts')
axc.set_title('chronometric')
axc.set_ylim([0.15,.8])
fig.suptitle('%s_%s_%s_%s' % (subject,model_name,type,refitted))

if to_save:
    savename = 'Visualisations/%s_%s_%s_%s.png' % (subject,model_name,type,refitted) 
    fig.savefig(basepath / savename,transparent=False,bbox_inches = "tight",format='png',dpi=300)

plt.show()
print('done')