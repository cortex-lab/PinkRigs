#%%
import pandas as pd

from dat_utils import get_paths
from inactivation_utils import fit_opto_model


_,formatted_path,savepath = get_paths(r'opto\region_comparison\uni')
all_files = list(formatted_path.glob('*SC.csv'))

#%%
results = []
for rec in all_files:
    brain_region = rec.stem
    trials = pd.read_csv(rec)



    subjects = trials.subject.unique()
    for subject in subjects:
        trials_of_subject = trials[trials.subject==subject]
        params,loss,auc = fit_opto_model(trials_of_subject,nametag=None,gammafit=False,L2opto=None)

        params['loss'] = loss
        params['auc'] = auc
        params['subject'] = subject
        params['brain_region'] = brain_region
        results.append(params)


results = pd.concat(results,ignore_index=True)
results = results.reset_index()



#
from plot_utils import plot_psychometric
#params,loss,auc = fit_opto_model(trials,nametag=None,gammafit=True,L2opto=None)

#
plot_psychometric(trials, gamma=1, weights = None,
                yscale='sig',ax=None, plot_curve = True, 
                colors=['b','grey','red'],
                dataplotkwargs={'marker':'o','ls':''},
                predpointkwargs ={'marker':'*','ls':''},
                predplotkwargs={'ls':'-'})
#results.to_csv(savepath / 'summary.csv')

# %%


import matplotlib.pyplot as plt
import seaborn as sns
# %%
import numpy as np
pivot_df = results.pivot(index = 'subject',columns = 'alpha',values = 'loss')
# %%
normalise_df = pivot_df.div(pivot_df[np.nan],axis=0)
# %%
sns.lineplot(normalise_df)

#%%
sns.lineplot(normalise_df.melt(),x='alpha',y='value')
plt.xscale('log')
# %%

sns.lineplot(data = results,
            x = 'alpha',y='visL_opto')
plt.xscale('log')

# %%
