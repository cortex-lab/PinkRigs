# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveSplit' 


import plots 
from preproc import read_pickle

#subjects = ['AV036','AV038','AV041','AV046','AV047']
subject = 'AV041_10mW_Bi'


refit_options = [
    'Ctrl',
    'starting_point', 
    'constant_bias',
    'sensory_drift', 
    'driftIC', 
    'all'
]

#for subject in subjects:
type = 'Ctrl'
refitted = refit_options[0]
plot_log = True
to_save = True

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\bilateral')
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

# data_path = basepath / ('Data/%s.csv' % subject)
# ev = pd.read_csv(data_path) 
# ev = preproc_ev(ev)
# Block = ev[~np.isnan(ev.rt_laserThresh) & ~ev.is_laserTrial.astype('bool')] 

model = read_pickle(model_path)
sample = read_pickle(sample_path)
actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))


# %% DIAGNOSTICS

fig,ax = plt.subplots(1,3,figsize=(24,5),sharey=True, sharex=True)
#
colors = ['b','k','r']
scaling_factor = 8
for ia,a in enumerate(actual_aud_azimuths):
    for i,v in enumerate(actual_vis_contrasts):
        curr_cond = conditions={'visDiff':v,'audDiff':a}
        plots.plot_diagnostics(model=model,sample = sample, 
                         conditions=curr_cond,data_dt =.025,method=None,myloc=i*scaling_factor,ax = ax[ia],
                         dkwargs={'color':colors[ia],'alpha':.5})

ax[0].set_ylim([.01,1.5])
ax[0].set_xticks(np.arange(actual_vis_contrasts.size)*scaling_factor)
ax[0].set_xticklabels(actual_vis_contrasts)
ax[0].set_xlabel('contrast')
ax[0].set_ylabel('reaction time (s)')
fig.suptitle('%s_%s'% (subject,refitted))
#ax[0].set_yscale('symlog')
# %% PSYCHOMETRIC
fig,(ax,axc) = plt.subplots(1,2,figsize=(20,10))
plots.plot_psychometric(model,sample,ax=ax,plot_log=False)
plots.plot_chronometric(model,sample,ax=axc)
fig.suptitle(refitted)
# %% CHRONOMETRIC 
model.parameters()
# %%
