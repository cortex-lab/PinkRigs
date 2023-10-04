# %%

# read in the the opto sample test set and compare the different models on it f
from pathlib import Path
from preproc import read_pickle
import re
import pyddm
import numpy as np
import pandas as pd

basepath = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\DriftAdditiveSplit')
sets = list(basepath.glob('*OptoSample_test.pickle'))

# %%
def get_logLik_helper(set):
    test_sample = read_pickle(set)
    namestubs = re.split('_',set.stem)
    setname =  '%s_%s_%s' % (tuple(namestubs[:3]))
    p = list(basepath.glob('%s_*Model*.pickle' % (setname))) 
    logLik,model_names = [], []
    for modelpath in p: 
        m = read_pickle(modelpath)    
        logLik.append(pyddm.get_model_loss(model=m,sample=test_sample))
        model_names.append(re.split('%s_' % (setname),modelpath.stem)[-1])

    return setname, model_names, np.array(logLik)[np.newaxis,:]

setname,model_names,logLiks= zip(*[get_logLik_helper(set) for set in sets])
logLiks = np.concatenate(logLiks)
df = pd.DataFrame(data=logLiks,columns=model_names[0],index=setname)

# %%
import matplotlib.pyplot as plt 
import seaborn as sns 


df_norm = df.sub(df['CtrlModel'],axis=0)
df_norm = df_norm.div(df['OptoModel_all']-df['CtrlModel'],axis=0)
# %%

_,ax = plt.subplots(1,1,figsize=(15,5))
ax.plot(df_norm.T)
ax.set_ylabel('normalised LogLikelihood')

# %%
