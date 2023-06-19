# %%
import sys
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
import pandas as pd
import numpy as np


from Analysis.neural.src.AV_model import AV_model,get_all_indicator_matrices
#from src.azimuthal_tuning import azimuthal_tuning
from Analysis.neural.utils.data_manager import load_cluster_info

rec_info = pd.Series(
    ['FT009','2021-01-20',7,'probe0'],
    index = ['subject','expDate','expNum','probe']
    )


# rec  = get_data_bunch('postactive')
# rec_info = rec.iloc[5]

# tuning_curve = azimuthal_tuning(rec_info=rec_info)
# tc = tuning_curve.get_tuning_curves(which = 'aud',cv_split=1)
# get the model fits 
av_model = AV_model('Linear')
av_model.load_data(**rec_info)
av_model.bin_kwargs['post_time'] = 0.5
av_model.format_spike_data()

IMs = get_all_indicator_matrices(
    av_model.additive_indicator_matrix,
    fit_type=av_model.regulariser,
    alpha=.0001
    )
repeats=1
scores = [av_model.fit_list_of_models(IMs, myrepeat) for myrepeat in range(repeats)]

model_names = list(scores[0].keys())
scores_all_reps=dict.fromkeys(model_names,[])
for m in model_names:
    meanscore = np.array([scores[i][m]  for i in range(len(scores))])
    scores_all_reps[m]  = meanscore 
    
meanVE  = np.array([scores_all_reps[m].mean(axis=0) for m in model_names])
meanVE = pd.DataFrame(meanVE.T,columns=model_names)
meanVE['winner_model'] = meanVE.idxmax(axis=1)
ve = meanVE.set_index(av_model.clusIDs,drop = True)
#%%
# get the cluster information. 
clusInfo = load_cluster_info(**rec_info)
rec_clusters_info = pd.concat([clusInfo,ve],axis = 1)
rec_clusters_info['recording_ID']  = '%s_%s_%s_%s' % tuple(rec_info)  # just a unique identification for the recordings's information. 
# %%
nrnID = 0


nrn_idx = np.where(rec_clusters_info.cluster_id==nrnID)[0][0]
av_model.fit_predict(IMs['V_spatial'])
av_model.plot_psths(nrn_idx,trainON=True,predON=False,testON=True,predcolor='green',predlw=1.2,fig=None)


# %%
import plotly.express as px

goodclus = rec_clusters_info[
   # (rec_clusters_info._av_KSLabels==2) & 
    (rec_clusters_info.depths.notna()) & 
    (rec_clusters_info._av_xpos.notna()) & 
    (rec_clusters_info.winner_model.notna()&
    (rec_clusters_info.spike_count>1000)) 
    ]
fig = px.scatter(
    goodclus,
    x='_av_xpos', y='depths',
    color='winner_model',
    hover_data=['cluster_id']
    )
fig.show()
# %%
sensory_clus = goodclus[goodclus.winner_model=='V_spatial']


fig = px.scatter(
    sensory_clus,
    x = 'ml', 
    y = 'ap',
    color = 'preferred_tuning'
)
fig.show()
# %%
# look at the cross_validated tuning curves for the spatial units: 
    #

tc_train = tc[tc.cv_number==0].reset_index()
tc_test = tc[tc.cv_number == 1 ].reset_index()
# select only the relevant units
tc_train = tc_train[clusInfo.winner_model=='V_spatial']
tc_test = tc_test[clusInfo.winner_model=='V_spatial']

# %%
import matplotlib.pyplot as plt
fig,ax  = plt.subplots(1,2,figsize = (15,8))
snake_train = tc_train.sort_values('preferred_tuning')
ax[0].imshow(snake_train.iloc[:,1:8],aspect = 'auto')
snake_test  = tc_test.loc[snake_train.index]
ax[1].imshow(snake_test.iloc[:,1:8],aspect = 'auto')

# %%
