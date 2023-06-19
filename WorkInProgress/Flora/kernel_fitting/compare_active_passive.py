
# %%
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import Bunch
from Analysis.pyutils.plotting import off_topspines,off_axes

from Analysis.neural.src.kernel_model import kernel_model

from kernel_params import get_params
dat_params,fit_params,eval_params = get_params()


dat_params_sess_type = Bunch({})

dat_params_sess_type['postactive'] = dat_params.copy()
dat_params_sess_type['postactive']['rt_params']= {'rt_min': None, 'rt_max': None}
dat_params_sess_type['postactive']['event_types'] = ['aud', 'vis', 'baseline']


dat_params_sess_type['multiSpaceWorld'] = dat_params.copy()
dat_params_sess_type['multiSpaceWorld']['rt_params'] = {'rt_min': 0.15, 'rt_max': 0.6}
dat_params_sess_type['multiSpaceWorld']['event_types'] = ['aud', 'vis', 'baseline','move']


results = Bunch({})
for sess in dat_params_sess_type.keys():
    curr_params = dat_params_sess_type[sess]
    kernels = kernel_model(t_bin=0.005,smoothing=0.025)
    kernels.load_and_format_data(
        subject = 'AV030',
        expDate = '2022-12-07', 
        expDef = sess,
        probe = 'probe0',
        subselect_neurons=None,
        **curr_params
    )
    # perform fitting
    kernels.fit(**fit_params)
    kernels.predict()
    variance_explained = kernels.evaluate(**eval_params)

    kernel_shapes = kernels.calculate_kernels()
    results[sess]= {
        've': variance_explained,
        'kernels': kernel_shapes,
        'clusIDs': kernels.clusIDs
        }

# %%
# possible areas of comparison: 
# kernel amplitude 
# kernel correlation
# variance explained per kernel? 


ve_P = results.postactive['ve']
ve_A = results.multiSpaceWorld['ve']

import matplotlib.pyplot as plt 

fig,ax = plt.subplots(1,1)

kernel_name = 'aud_kernel_spl_0.10_azimuth_60'
nrnID= 330
for sess in dat_params_sess_type.keys():
    nrnIdx = np.where(results[sess]['clusIDs']==nrnID)[0][0]
    ax.plot(
    results[sess]['kernels'][kernel_name][nrnIdx,:]
    )
    # ax.plot(
    # results[sess]['kernels']['vis_kernel_contrast_0.20_azimuth_60'][nrnIdx,:],
    # color='g'

    # )

    # ax.plot(
    # results[sess]['kernels']['vis_kernel_contrast_0.10_azimuth_60'][nrnIdx,:],
    #     color='b'
    # )


ax.legend(list(dat_params_sess_type.keys()))

print(ve_A[(ve_A.clusID==nrnID)])

# %%

# calculate the angle between active and passive kernels
import pandas as pd
kernel_names = list(results.postactive['kernels'].keys())

cIDs,cIdxP,cIdxA = np.intersect1d(
    results.postactive['clusIDs'],
    results.multiSpaceWorld['clusIDs'],return_indices=True
)


similarity = {}



audP = [ve_P[(ve_P.clusID==nrnID) & (ve_P.cv_number==1) & (ve_P.event=='aud')].VE.values[0] for nrnID in cIDs]
visP = [ve_P[(ve_P.clusID==nrnID) & (ve_P.cv_number==1) & (ve_P.event=='vis')].VE.values[0] for nrnID in cIDs]


audA = [ve_A[(ve_A.clusID==nrnID) & (ve_A.cv_number==1) & (ve_A.event=='aud')].VE.values[0] for nrnID in cIDs]
visA = [ve_A[(ve_A.clusID==nrnID) & (ve_A.cv_number==1) & (ve_A.event=='vis')].VE.values[0] for nrnID in cIDs]

for k in kernel_names[:-1]:
    slopes = []
    for nrnID in cIDs:

        # look at how much varaince was explained by the aud/vis kernels for these neurons


        # plot the VE against each other? 

        nrnIdxP = np.where(results['postactive']['clusIDs']==nrnID)[0][0]
        a = results['postactive']['kernels'][k][nrnIdxP,20:60]
        nrnIdxA = np.where(results['multiSpaceWorld']['clusIDs']==nrnID)[0][0]
        b = results['multiSpaceWorld']['kernels'][k][nrnIdxA,20:60]

        covmat = np.cov([a,b])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:,largest_index]

        slope = np.rad2deg(np.arctan([largest_eig_vec[1]/largest_eig_vec[0]]))[0]
        slopes.append(slope)
    similarity[k] = slopes

similarity = pd.DataFrame.from_dict(similarity)

similarity['ve_aud_passive'] = audP
similarity['ve_vis_passive'] = visP
similarity['ve_aud_active'] = audA
similarity['ve_vis_active'] = visA


# %%

sim_vis = similarity[
    ((similarity.ve_vis_active>0.02) + (similarity.ve_vis_passive>0.02))

    & 
    (similarity.ve_vis_active>-.5)
]

# %%
import seaborn as sns
sns.pairplot(sim_vis,diag_kws={'bins':20})

# %%
# %%


# %%
nrnID=59

nrnIdxP = np.where(results['postactive']['clusIDs']==nrnID)[0][0]
a = results['postactive']['kernels'][k][nrnIdxP,20:60]
nrnIdxA = np.where(results['multiSpaceWorld']['clusIDs']==nrnID)[0][0]
b = results['multiSpaceWorld']['kernels'][k][nrnIdxA,20:60]

covmat = np.cov([a,b])
print(covmat)
eig_values, eig_vecs = np.linalg.eig(covmat)
largest_index = np.argmax(eig_values)
largest_eig_vec = eig_vecs[:,largest_index]
fig,ax = plt.subplots(1,1)
ax.plot(a,b,'.')
ax.plot([0,.25],[0,.25],'k--')

divisor=3
ax.plot([0,largest_eig_vec[0]/divisor],[0,largest_eig_vec[1]/divisor])

print(np.rad2deg(np.arctan([largest_eig_vec[1]/largest_eig_vec[0]])))


# %%

import matplotlib.pyplot as plt
cIDs,cIdxP,cIdxA = np.intersect1d(
    results.postactive['clusIDs'],
    results.multiSpaceWorld['clusIDs'],return_indices=True
)

cIdx = [cIdxP,cIdxA]

res = []

kernel_names = list(results.postactive['kernels'].keys())


fig, ax = plt.subplots(len(kernel_names),1,figsize=(2,20))

for idx,k in enumerate(kernel_names):

    res_dict = []
    sess_types = list(dat_params_sess_type.keys())
    for i,sess in enumerate(sess_types):
        my_kernels = results[sess]['kernels']
        res_dict.append(np.ptp(my_kernels[k][cIdx[i],:],axis=1))

    ax[idx].plot(res_dict[0],res_dict[1],'o')
    ax[idx].set_xlabel(sess_types[0])
    ax[idx].set_ylabel(sess_types[1])
    ax[idx].set_title(k)
    ax[idx].plot([0,.25],[0,.25],'k--')
# %%

import seaborn as sns


sns.pairplot(res, hue="sess_type")

# %%
my_diff = (np.ptp(results.multiSpaceWorld['kernels']['vis_kernel_contrast_0.20_azimuth_60'][cIdxA,:],axis=1) -
np.ptp(results.postactive['kernels']['vis_kernel_contrast_0.20_azimuth_60'][cIdxP,:],axis=1)) 

cIDs[np.argsort(my_diff)]



#%%
my_diff[cIDs==nrnID]

# %%
