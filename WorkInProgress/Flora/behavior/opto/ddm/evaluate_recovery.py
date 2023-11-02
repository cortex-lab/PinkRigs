"""
This is code to assess how the starting point vs constant bias models really differ

"""
# %% 


import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle

#for subject in subjects:

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
model_path  = basepath / 'ClusterResults/resampled_fits'
sample_path  = basepath / 'Data/forMyriad/resamples'
sample_path_train = sample_path / 'train'
sample_path_test = sample_path / 'test'

# winner model on the test set
def get_all_models_LogLik(dataset):
    """
    local helper to get the performance of all models on a particular dataset 
    Parameters: 
    dataset: pathlib.Path 
    """
    test_path = sample_path_test / ('%s.pickle' % (dataset.stem))
    test_sample = read_pickle(test_path)

    # get which models to evaluate 
    model_paths = list(model_path.glob('%s_*.pickle' % dataset.stem))
    model_names = [re.split('_Model_*',path.stem)[-1] for path in model_paths]

    LogLiks_train,LogLiks_test = [],[]
    d_b,d_x0 = [], []
    for path in model_paths: 
        model = read_pickle(path)
        LogLiks_train.append(model.fitresult.value())
        LogLiks_test.append(pyddm.get_model_loss(model=model,sample=test_sample))
        d_b.append(model.parameters()['drift']['d_b'].real)
        d_x0.append(model.parameters()['IC']['d_x0'].real)
    
    LogLiks_train = np.array(LogLiks_train)[np.newaxis,:]
    LogLiks_test = np.array(LogLiks_test)[np.newaxis,:]
    d_b = np.array(d_b)[np.newaxis,:]
    d_x0 = np.array(d_x0)[np.newaxis,:]

    return model_names,LogLiks_train,LogLiks_test,d_b,d_x0


assesed_models =[
    'g_d_b',
    'g_d_x0',
    'g_both'

]

LogLik_per_simulated,LogLik_per_simulated_ = [],[]
d_b_,d_x0_ = [],[]
for model_name in assesed_models:
    datasets = list(sample_path_train.glob('*%s*.pickle' % model_name))
    mname,LogLik,LogLik_,d_b,d_x0 = zip(*[get_all_models_LogLik(dataset) for dataset in datasets])
    LogLik, LogLik_ = np.concatenate(LogLik),np.concatenate(LogLik_)
    d_b,d_x0 = np.concatenate(d_b),np.concatenate(d_x0)
    LogLik_per_simulated.append(LogLik)
    LogLik_per_simulated_.append(LogLik_)
    d_b_.append(d_b)
    d_x0_.append(d_x0)
mname = mname[0]

# %%
# construct the confusion matrix for the test set
fit_models = np.array(mname[-3:])
n_sim = len(assesed_models)
n_fit = fit_models.size
confusion_matrix = np.zeros((n_sim,n_fit))
for i in range(n_sim):
    ll_m = LogLik_per_simulated_[i]
    winner_idx = np.argmin(LogLik_per_simulated_[i][:,-n_fit:],axis=1)
    for j in range(n_fit):
        confusion_matrix[i,j] = np.mean(winner_idx==j)
# confusion mmatrix, test set 

fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.matshow(confusion_matrix,cmap='viridis',vmin=0,vmax=1)
for (i, j), z in np.ndenumerate(confusion_matrix):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

ax.set_xticks(np.arange(0,n_fit))
ax.set_yticks(np.arange(0,n_sim))
ax.set_xticklabels(fit_models)
ax.set_yticklabels(assesed_models)
ax.set_xlabel('fit model')
ax.set_ylabel('simulated model')
ax.set_title('confusion matrix')
# %% look at how the two parameters trade against if at all

# but this is the best to look in the simualtion
fig,ax = plt.subplots(1,n_sim,figsize=(8,3),sharex=True,sharey=True)
fig.suptitle('delta parameters when g_both is fitted')
fig.tight_layout()

for i,m in enumerate(assesed_models):
    x = d_b_[i][:,2]
    y = d_x0_[i][:,2]
    ax[i].plot(
        x,
        y,'o',markerfacecolor='cyan',markeredgecolor='k'

    )
    ax[i].set_title('sim: %s, r =%.2f' % (m,np.corrcoef(x,y)[0,1]))

ax[0].set_xlabel('drift bias')
ax[0].set_ylabel('starting point')
# %%

idx = np.argmin(LogLik_per_simulated_[2][:,-n_fit:],axis=1)==1
plt.hist(d_x0_[2][~idx,2],bins=50,alpha=.7)
plt.hist(d_x0_[2][idx,2],bins=10,alpha=.7)
plt.xlabel('d_x0')
plt.title('what is d_x0 when g_d_b on a g_both simulation?')
plt.legend(['g_both wins','g_d_b wins'])
# %%
