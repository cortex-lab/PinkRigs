# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
model_name = 'DriftAdditiveOpto' 
import pyddm
from pyddm.plot import model_gui as gui
import plots 
from preproc import read_pickle



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

# this load will only work if the model function is in path...
basepath = Path(r'C:\Users\Flora\Documents\ProcessedData\ddm\Opto')
model_path  = basepath / 'ClusterResults/DriftAdditiveOpto'
sample_path = basepath / 'Data/forMyriad/samples/train'

datasets = list(sample_path.glob('*.pickle'))


dataset = datasets[0]

def get_dvals(dataset,type='all'):
    model = read_pickle(model_path / ('%s_Model_%s.pickle' % (dataset.stem,type)))
    p = model.parameters()
    needed_keys = []
    for k in p.keys():
        cd = p[k]
        for ck in cd.keys():
            needed_keys.append((k,ck))

    p = {k[1]:p[k[0]][k[1]].real for k in needed_keys}

    ctrl = {
        'vR': p['v'],
        'vL': p['v'] + p['vS'],       
        'aR': p['a'],
        'aL': p['a'] + p['aS'],         
        'b': p['b'],
        'x0': p['x0'],
        'B':p['B'],
        'mixture': p['pmixturecoef'],
        'nondectime': p['nondectime']

    }

    opto = {
        'vR': p['v'] + p['d_vR'],
        'vL': p['v'] + p['vS'] + p['d_vR'],       
        'aR': p['a'] + p['d_aR'] ,
        'aL': p['a'] + p['aS'] + p['d_aL'],         
        'b': p['b'] + p['d_b'],
        'x0': p['x0']+ p['d_x0'],
        'B': p['B'] + p['d_B'],
        'mixture': p['pmixturecoef'] + p['d_pmixturecoef'],
        'nondectime': p['nondectime'] + p['d_nondectimeOpto']

    }

    ctrlvalues = (np.array(list(ctrl.items()))[:,1]).astype('float')
    optovalues = (np.array(list(opto.items()))[:,1]).astype('float')

    return list(opto.keys()),ctrlvalues[np.newaxis,:],optovalues[np.newaxis,:]

which_model = 'g_boundx0'
namekeys,ctrl,opto = zip(*[get_dvals(dataset,type=which_model) for dataset in datasets])
namekeys,ctrl,opto = namekeys[0],np.concatenate(ctrl).T,np.concatenate(opto).T
#%%
n = len(namekeys)
ss = False
fig,ax = plt.subplots(1,n,figsize=(20,2),sharey=ss,sharex=ss)
fig.tight_layout()
for i,(c,o,a,t) in enumerate(zip(ctrl,opto,ax,namekeys)):
    a.plot(c,o,'o')
    a.set_title(t)
    lim = np.max(np.abs(np.concatenate((c,o)))) *1.1
    a.plot([-lim,lim],[-lim,lim],'k--')

ax[0].set_xlabel('control')
ax[0].set_ylabel('opto')
fig.suptitle('Model params from model  %s' % which_model)
# %%

namekeys = np.array(namekeys)
plt.rcParams.update({'font.size': 22})

fig,ax = plt.subplots(1,1,figsize=(5,5))
my_b = np.ravel(opto[namekeys=='b',:])
my_x0 = np.ravel(opto[namekeys=='x0',:])
ax.plot(
    my_b,
    my_x0,'o'
    
)
ax.set_xlabel('drift bias')
ax.set_ylabel('starting point')
ax.set_title(' Model: %s,r=%.2f' % (which_model,np.corrcoef(my_b,my_x0)[0,1]))

# %%

fig,ax = plt.subplots(1,1,figsize=(5,5))
my_b = np.ravel(opto[namekeys=='x0',:]) -  np.ravel(ctrl[namekeys=='x0',:])
my_x0 = np.ravel(opto[namekeys=='B',:])
ax.plot(
    my_b,
    my_x0,'o'
    
)
ax.set_xlabel('delta starting point')
ax.set_ylabel('bound')
ax.set_title(' Model: %s,r=%.2f' % (which_model,np.corrcoef(my_b,my_x0)[0,1]))

# %%
