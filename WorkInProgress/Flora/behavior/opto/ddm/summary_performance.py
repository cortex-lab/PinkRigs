# for each session we plot the total logLikelihood on the  training vs test set 

# %%
import re,glob,sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveOpto' 
import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle

#for subject in subjects:

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'
sample_path  = basepath / 'Data/forMyriad/samples'
sample_path_train = sample_path / 'train'
sample_path_test = sample_path / 'test'
datasets = list(sample_path_train.glob('*.pickle'))


pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Analysis.pyutils.plotting import off_axes,off_topspines


# get each model's performance 
# %%
#dataset = datasets[8]

LogLik_train_all,LogLik_test_all,fitted_models = [], [], []
for dataset in datasets:
    model_paths = list(model_path.glob('%s_*.pickle' % dataset.stem))
    model_names = [re.split('_Model_*',path.stem)[-1] for path in model_paths]
    fitted_models.append(model_names)
    toplot=True
    LogLiks_train,LogLiks_test = [],[]
    for path in model_paths: 
        model = read_pickle(path)
        LogLiks_train.append(model.fitresult.value())
        dataset_name = re.split('_train*',path.stem)[0]
        test_path = sample_path_test / ('%s_test.pickle' % (dataset_name))
        test_sample = read_pickle(test_path)
        LogLiks_test.append(pyddm.get_model_loss(model=model,sample=test_sample))

    if toplot:
        fig,ax = plt.subplots(1,1,figsize=(10,5),sharex=True)
        ax.plot(LogLiks_train,model_names,'o-')
        ax.plot(LogLiks_test,model_names,'o-')
        ax.legend(['train','test'])

        fig.suptitle(dataset_name)
        titles = ['train','test']
            # a.set_xticks(np.arange(len(LogLiks_test)))
            # a.set_xticklabels(model_names,rotation=90)
       # ax.set_title(t)

        ax.set_xlabel('-LogLikelihood')

    LogLik_train_all.append(np.array(LogLiks_train)[np.newaxis,:])
    LogLik_test_all.append(np.array(LogLiks_test)[np.newaxis,:])

LogLik_train_all = np.concatenate(LogLik_train_all)
LogLik_test_all = np.concatenate(LogLik_test_all)


# %%

# how does all compare to single variables of intersest e.g. 


# %%

fig,ax= plt.subplots(1,1)
ax.plot(model_names,LogLik_train_all.T)

# %%

train_norm = (LogLik_train_all-LogLik_train_all[:,0][:,np.newaxis])/(LogLik_train_all[:,1]-LogLik_train_all[:,0])[:,np.newaxis]

test_norm = (LogLik_test_all-LogLik_test_all[:,0][:,np.newaxis])/(LogLik_test_all[:,1]-LogLik_test_all[:,0])[:,np.newaxis]

# %%
# plot a bunch of losst performances against each other 

wh = [14,15,17,9,10,12,13,16,11]
model_names[wh]
#%%
fig,ax= plt.subplots(1,1,figsize=(20,4))
ax.plot(model_names,test_norm.T)
# %%
wh = [2,14,3,4,13,15]

wh = [3,4,13,15]
wh = [5,6,7,8,9,10,11,12]
wh = [5,10,6,11,12,7]

wh = [0,11,16]

wh = [5,3,4,2]
wh = [14,15,17,9,10,12,13,16,11]

model_names = np.array(model_names)
fig,ax= plt.subplots(1,1)
ax.set_title('train')
ax.plot(1-train_norm.T[wh,:],'k',alpha=.3)
ax.set_xticklabels(model_names[wh],rotation=45)
ax.axhline(0,color='r')

# %%
fig,ax= plt.subplots(1,1)
ax.set_title('test')
ax.plot(1-test_norm.T[wh,:],'k',alpha=.3)
ax.set_xticks(np.arange(len(wh)))
ax.set_xticklabels(model_names[wh],rotation=45)
ax.axhline(1,color='r')


mypath = r'C:\Users\Flora\Pictures\SfN2023'
savename = mypath + '\\' + 'loss_models_test_set.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)


# %%

wh = ['d_x0','d_b']
n = len(wh)
size = 2.5
fig,ax = plt.subplots(1,n,figsize=(size*n,size),sharex=True,sharey=True)

dl=.15
for i,m in enumerate(wh):
    ax[i].scatter(
        1-np.ravel(test_norm.T[model_names==('g_%s' % m),:]),
        np.ravel(test_norm.T[model_names==('l_%s' % m),:]),
        s=72,facecolors='cyan',edgecolors='k',alpha=1,linewidth=2) 

    ax[i].set_title(m)
    off_axes(ax[i])
    ax[i].vlines(0,-dl,1+dl,color='k')
    ax[i].hlines(0,-dl,1+dl,color='k')
    ax[i].vlines(1,-dl,0,color='k')
    ax[i].hlines(1,-dl,0,color='k')





mypath = r'C:\Users\Flora\Pictures\SfN2023'
savename = mypath + '\\' + 'gainloss.svg'
fig.savefig(savename,transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
