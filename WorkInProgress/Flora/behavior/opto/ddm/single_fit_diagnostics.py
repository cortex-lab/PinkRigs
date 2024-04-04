# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

model_name = 'DriftAdditiveOpto' 
import pyddm,glob,sys,re
from pathlib import Path
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle

pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Analysis.pyutils.plotting import off_axes,off_topspines

#subjects = ['AV036','AV038','AV041','AV046','AV047'

refit_options = [
    'all',
    'l_aS',
    'l_vS',
    'l_d_mixturecoef',
    'g_d_b',
    'g_both', 
    'g_boundx0',
    'g_d_x0',
    'l_v', 
    ]

#for subject in subjects:

type = refit_options[7]
plot_log = True
to_save = True

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')

#
subjects = list((basepath/'ClusterResults/DriftAdditiveOpto').glob('*Model_all.pickle'))
subjects = [re.findall(r'(\w+)_Sample_train_Model_all', s.stem)[0] for s in subjects]
#subjects = ['AV041_10mW_Left']
for subject in subjects:
    model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'

    #model_path = Path(r'C:\Users\Flora\Documents\Github\PinkRigs\WorkInProgress\Flora\behavior\opto\ddm\DriftAdditiveOpto')

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
    #
    print('LogLik',type,model.fitresult.value())
    #  DIAGNOSTICS

    fig,ax = plt.subplots(2,3,figsize=(15,4.5),sharey=True, sharex=True)
    plt.rcParams['font.size'] = 15

    colors = ['b','k','r']
    scaling_factor = 8
    for isLaser in range(2):
        for ia,a in enumerate(actual_aud_azimuths):
            for i,v in enumerate(actual_vis_contrasts):
                curr_cond = conditions={'visDiff':v,'audDiff':a,'is_laserTrial':isLaser}
                plots.plot_diagnostics(model=model,sample = sample, 
                                conditions=curr_cond,data_dt =.025,method=None,myloc=i*scaling_factor,ax = ax[isLaser,ia],
                                dkwargs={'color':colors[ia],'alpha':.5})
                off_topspines(ax[isLaser,ia])

    ax[0,0].set_ylim([.01,1.5])
    ax[0,0].set_title('ctrl trials',loc='left')
    ax[1,0].set_title('opto trials',loc='left')
    ax[0,0].set_xticks(np.arange(actual_vis_contrasts.size)*scaling_factor)
    ax[0,0].set_xticklabels(actual_vis_contrasts)
    ax[1,0].set_xlabel('contrast')
    ax[0,0].set_ylabel('reaction time (s)')
    ax[0,0].set_ylim([.28,.7])
    fig.suptitle('%s_%s'% (subject,type))

    mypath = r'C:\Users\Flora\Pictures\SfN2023'
    savename = mypath + '\\' + 'diagnostics_%s_%s.svg' % (subject,type)
    fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
# #ax[0].set_yscale('symlog')
# %% PSYCHOMETRIC

fig,ap = plt.subplots(1,2,figsize=(6,3),sharey=True)
plots.plot_psychometric(model,sample,axctrl=ap[0],axopto=ap[1],plot_log=True)
fig.suptitle('psychometric')
off_topspines(ap[0])
off_topspines(ap[1])
savename = mypath + '\\' + 'ps_%s.svg' % subject
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
fig,ac = plt.subplots(1,2,figsize=(6,3),sharey=True)
wh = 'all'
plots.plot_chronometric(model,sample,axctrl=ac[0],axopto=ac[1],which=wh,metric_type='median')

off_topspines(ac[0])
off_topspines(ac[1])
savename = mypath + '\\' + 'chrono_%s.svg' % subject
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)
fig.suptitle('%s choices' % wh)
# fig,ac = plt.subplots(1,2,figsize=(6,3),sharey=True)
# plots.plot_chronometric(model,sample,axctrl=ac[0],axopto=ac[1],which='right')
# fig.suptitle('right choices')
#%%
#ac[1].set_ylim(ac[0].get_ylim())
# fig.suptitle(refitted)

# %%
