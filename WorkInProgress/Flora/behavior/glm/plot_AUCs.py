# %%


from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score

from predChoice_utils import fit_stim_vs_neur_models
from plot_utils import copy_svg_to_clipboard


def plot_AUCs(logOdds_stim,logOdds_neur,choices):
    '''
    this function plots the stim only vs all the predictors odds
    '''

    df = pd.DataFrame({
        'LogOdds,stim': logOdds_stim,
        'LogOdds,neural': logOdds_neur,
        'choice':choices,
    })

    jointdat = np.array([logOdds_stim,logOdds_neur])

    g= sns.jointplot(data=df,
                x="LogOdds,stim", y="LogOdds,neural", hue='choice',
                palette='coolwarm',hue_norm=(.1,.9),
                kind='scatter',ratio=4,edgecolor='k',
                xlim=[np.min(jointdat)*1.05,np.max(jointdat)*1.05],
                ylim=[np.min(jointdat)*1.05,np.max(jointdat)*1.05],
                legend=False)


    g.ax_joint.axline((0,0),slope=1,color='k',linestyle='--')
    g.ax_joint.axvline(0,color='k',linestyle='--')
    g.ax_joint.axhline(0,color='k',linestyle='--')
    g.ax_marg_x.set_title('AUC = %.2f' % 
                        roc_auc_score(df['choice'].values,logOdds_stim))

    g.ax_marg_y.set_title('AUC = %.2f' % 
                        roc_auc_score(df['choice'].values,logOdds_neur))
    g.ax_marg_x.legend(['right choice','left choice'])

    plt.show()

    return g 




from itertools import product

brain_areas = ['SCm']
paramsets = ['choice']


to_compare = ['stim','all']


res = []
for x in product(brain_areas,paramsets):
    region,tt = x[0],x[1]
    savepath = Path(r'D:\LogRegression\%s_%s\formatted' % (region,tt))
    all_files = list(savepath.glob('AV030*.csv'))

    for rec in all_files:
        trials = pd.read_csv(rec)
        results = fit_stim_vs_neur_models(trials,neuron_selector_type='lasso')

        models_  = [results['models'][model] for model in to_compare]
        Xs_  = [results['X_test_subsets'][model] for model in to_compare]
        y = results['y_test']


        LogOdds = [model.decision_function(X) for model,X in zip(models_,Xs_)]
        model1_Odds,model2_Odds =LogOdds[0],LogOdds[1]

        out = pd.DataFrame({
            'Odds_stim': model1_Odds, 
            'Odds_all':model2_Odds,
            'choices': y
        })

        res.append(out)


res = pd.concat(res)


#%%
fig = plot_AUCs(res.Odds_stim,res.Odds_all,res.choices)
copy_svg_to_clipboard(fig)

# %%
