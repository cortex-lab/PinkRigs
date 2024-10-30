#%% 
# this script is used to asses the neurometric log-Odds model performance using simulated neurons

import datetime
import time

# generic libs 
import numpy as np
import pandas as pd


from pathlib import Path 

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns


from predChoice_utils import make_fake_data,fit_parse_



rec = Path(r'D:\LogRegression\SCm_poststim\formatted\AV008_2022-03-10_1.csv')

trials = pd.read_csv(rec)


# simulation no1

# Q1: how many initial neurons

init_nrns = [6,10,15]
results = []
for n in init_nrns:
    t0 = time.time()

    trials = make_fake_data(trials, 
                n_fake_neurons = n,
                p_choice_neurons = 0,
                p_vis_neurons = .4,
                p_aud_neurons = .4,
                p_av_neurons = 0,
                p_v_choice_neurons = 0,
                p_a_choice_neurons = 0,
                p_av_choice_neurons = 0,
                noise_sd = .2)

    df = fit_parse_(trials,nametag=None,
                    neuron_selector_type='lasso')
                    
    df['fit_time'] = time.time()-t0
    results.append(df)

results = pd.concat(results)


# %%
import seaborn as sns
sns.lineplot(data=results,
            x='nNeur',y='fit_time')

# %%
dd = results.filter(like='roc_auc_score_test').T
dd.columns = results.nNeur

sns.lineplot(dd)
plt.xticks(rotation=45)
#plt.ylim([0.2,1.1])
# %%
df
# %%
