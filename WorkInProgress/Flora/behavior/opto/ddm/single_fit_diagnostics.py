# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveOpto' 
import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle

#subjects = ['AV036','AV038','AV041','AV046','AV047']
subject = 'AV044_10mW_Right'


refit_options = [
        'ctrl',
        'drift_bias',
        'sensory_drift',
        'starting_point',
        'mixture',
        'nondectime',
        'all'
    ]

#for subject in subjects:
type = refit_options[6]
plot_log = True
to_save = True

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'
sample_path = basepath / 'Data/forMyriad/samples/train'

sample_path = (sample_path / ('%s_Sample_train.pickle' % (subject)))


model_path = (model_path / ('%s_Model_%s.pickle' % (sample_path.stem,type)))


# data_path = basepath / ('Data/%s.csv' % subject)
# ev = pd.read_csv(data_path) 
# ev = preproc_ev(ev)
# Block = ev[~np.isnan(ev.rt_laserThresh) & ~ev.is_laserTrial.astype('bool')] 

model = read_pickle(model_path)
sample = read_pickle(sample_path)
actual_aud_azimuths = np.sort(np.unique(sample.conditions['audDiff'][0]))
actual_vis_contrasts =  np.sort(np.unique(sample.conditions['visDiff'][0]))
model.parameters()
#%%
#gui(model=model, sample=sample, conditions={"audDiff": actual_aud_azimuths, "visDiff": actual_vis_contrasts,"is_laserTrial":[False,True]})

#%%
print('LogLik',type,model.fitresult.value())
# %% DIAGNOSTICS

fig,ax = plt.subplots(2,3,figsize=(24,5),sharey=True, sharex=True)
#
colors = ['b','k','r']
scaling_factor = 8
for isLaser in range(2):
    for ia,a in enumerate(actual_aud_azimuths):
        for i,v in enumerate(actual_vis_contrasts):
            curr_cond = conditions={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
            plots.plot_diagnostics(model=model,sample = sample, 
                            conditions=curr_cond,data_dt =.025,method=None,myloc=i*scaling_factor,ax = ax[isLaser,ia],
                            dkwargs={'color':colors[ia],'alpha':.5})

ax[0,0].set_ylim([.01,1.5])
ax[0,0].set_title('ctrl trials',loc='left')
ax[1,0].set_title('opto trials',loc='left')
ax[0,0].set_xticks(np.arange(actual_vis_contrasts.size)*scaling_factor)
ax[0,0].set_xticklabels(actual_vis_contrasts)
ax[1,1].set_xlabel('contrast')
ax[0,0].set_ylabel('reaction time (s)')
fig.suptitle('%s_%s'% (subject,type))
# #ax[0].set_yscale('symlog')
# %% PSYCHOMETRIC

fig,(ap,ac) = plt.subplots(2,2,figsize=(10,10))

plots.plot_psychometric(model,sample,axctrl=ap[0],axopto=ap[1],plot_log=False)
plots.plot_chronometric(model,sample,axctrl=ac[0],axopto=ac[1])
#%%
#ac[1].set_ylim(ac[0].get_ylim())
# fig.suptitle(refitted)

# %%
