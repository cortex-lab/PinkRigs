#  %%
import sys
from turtle import color
import pandas as pd
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.batch_data import get_data_bunch
from pathlib import Path
from Admin.csv_queryExp import load_data

# visualisations
import seaborn as sns
import matplotlib.pyplot as plt

save_path = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')

dataset = 'naive-allen'
fit_tag = 'additive-fit'

interim_data_folder = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual')
save_path = interim_data_folder / dataset / 'kernel_model' / fit_tag

sc_probeloc_path = Path(r'C:\Users\Flora\Documents\Processed data\passiveAV_project')

recordings = get_data_bunch(dataset)

cluster_info = list()
for _,rec_info in recordings.iterrows():
    print('Loading %s %s, expNum = %.0f, %s' % tuple(rec_info))
    probe = rec_info.probe
    raw_probe = ('%s_raw' % probe)
    data_dict = {probe:{'clusters':'all'},raw_probe:{'clusters':'all'}}
    sess = load_data(**(rec_info.iloc[:3]),data_name_dict=data_dict)
    fit_results = pd.read_csv((save_path / ('%s_%s_%.0f_%s.csv' % tuple(rec_info))))
    clus = sess[probe].iloc[0].clusters
    for k in clus.keys():
        fit_results[k] = [clus[k][np.where(clus.cluster_id==c)[0][0]] for c in fit_results.clusID]
    clus_raw = sess[raw_probe].iloc[0].clusters
    for k in clus_raw.keys():
        fit_results[('r_'+ k)] = [clus_raw[k][np.where(clus.cluster_id==c)[0][0]] for c in fit_results.clusID]

    #now we need to find the clusters shank postiitons
    print('aaaaaa')
    probe_imec = 'imec0' if 'probe0' in rec_info.probe else 'imec1'
    registration_folder = sc_probeloc_path / rec_info.subject / rec_info.expDate / 'alf' / probe_imec
    registration_files = list(registration_folder.glob('*.npy')) 

    if len(registration_files)==4:   
        d = {}
        for i,r in enumerate(registration_files):
            d[i] = np.load(r)

        # and now assign each for the unit. 
        fit_results['sc_azimuth'] = [d[s][0] for s in fit_results._av_shankID]
        fit_results['sc_elevation'] = [d[s][1] for s in fit_results._av_shankID]

    # add all the recording specific information
    fit_results['subject'] = rec_info.subject
    fit_results['expDate'] = rec_info.expDate
    fit_results['probe']= probe
    fit_results['expNum'] =rec_info.expNum
    fit_results['session_ID'] = '%s_%s_%.0f_%s' % tuple(rec_info)

    cluster_info.append(fit_results)

cluster_info = pd.concat(cluster_info)

# fantastic metadata output
# %%
# plot pairplot for ve 

plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':36})
plt.rcParams['figure.dpi'] = 300
cluster_info_test = cluster_info[(cluster_info.cv_number==1) & (cluster_info._av_KSLabels==2)]
df = cluster_info_test[['VE','neuron','event','session_ID']]
df = df.reset_index(drop=True)
df = df.pivot(index=['neuron','session_ID'],columns='event')
# %%
sns.pairplot(df.VE[['aud','vis','non-linearity']],plot_kws=dict(marker="o",alpha=.7))
# %%
def get_my_acronym(allen_acronym):    
    if ('SCs' in allen_acronym) or ('SCo' in allen_acronym) or ('SCzo' in allen_acronym):
        my_acronym = 'SCs'
    elif ('SCi' in allen_acronym):
        my_acronym = 'SCi'
    elif ('SCd' in allen_acronym):
        my_acronym = 'SCd'
    else:
        my_acronym = 'midbrain'
    
    return my_acronym
        

# %% specific to brain regions
anat_done =cluster_info_test[cluster_info_test.r_brainLocationAcronyms_ccf_2017.notna()]
anat_done['location'] = [get_my_acronym(c)  for c in anat_done.r_brainLocationAcronyms_ccf_2017.values]
anat_done = anat_done[['VE','neuron','event','session_ID','location']]
anat_done = anat_done.reset_index(drop=True)
anat_done = anat_done.pivot(index=['session_ID','neuron','location'],columns=['event'])
anat_done=anat_done.VE
anat_done = anat_done.reset_index()
# %%
sns.pairplot(
    anat_done,vars=['vis','aud','non-linearity'],hue='location',
    kind='scatter',plot_kws=dict(marker="o", alpha=.7,s=24)
    )

# %%
sc_done =cluster_info_test[cluster_info_test.sc_azimuth.notna()]
sc_done['d_from_sc_surface'] = sc_done.depths - sc_done.sc_surface
sc_done = sc_done[['VE','neuron','event','session_ID','d_from_sc_surface','sc_azimuth','sc_elevation']]

sc_done = sc_done.pivot(index=['session_ID','neuron','d_from_sc_surface','sc_azimuth','sc_elevation'],columns=['event'])
sc_done = sc_done.VE
sc_done = sc_done.reset_index()

import matplotlib.pyplot as plt 
_,ax = plt.subplots(1,3,figsize=(9,4),sharey=True)
thr = .02
#pp=sns.histplot(sc_done[sc_done.vis>thr],x='vis',y='d_from_sc_surface',ax=ax[0],color='blue',bins=10)
#sns.histplot(sc_done[sc_done.aud>thr],x='aud',y='d_from_sc_surface',ax=ax[1],color='magenta',bins=10)
#sns.histplot(sc_done[sc_done.motionEnergy>thr],x='motionEnergy',y='d_from_sc_surface',ax=ax[2],color='black')

pp=sns.histplot(sc_done,x='vis',y='d_from_sc_surface',ax=ax[0],color='blue',bins=20)
sns.histplot(sc_done,x='aud',y='d_from_sc_surface',ax=ax[1],color='magenta',bins=20)
sns.histplot(sc_done,x='non-linearity',y='d_from_sc_surface',ax=ax[2],color='green',bins=20)

ax[0].set_ylim([-1500,100])
from Analysis.pyutils.plotting import off_exceptx, off_topspines
off_exceptx(ax[1])
off_exceptx(ax[2])
off_exceptx(ax[0])
plt.savefig("C:\\Users\\Flora\\Pictures\\depthplots_nl.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%

plt.scatter(sc_done.aud,sc_done.d_from_sc_surface,color='m')
plt.scatter(sc_done.vis,sc_done.d_from_sc_surface,color='b')
# %%
plt.scatter(sc_done['non-linearity'],sc_done.d_from_sc_surface,color='g')
plt.scatter(sc_done['baseline'],sc_done.d_from_sc_surface,color='k')

# %%
 

# %%
sc_ = sc_done[(sc_done.d_from_sc_surface<0) & (sc_done.d_from_sc_surface>-1400)]


fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
ax[0].plot(sc_.vis.values,sc_.aud.values,'o',alpha=0.2,color='k')
ax[1].plot(sc_.motionEnergy.values,sc_.aud.values,'o',alpha=0.2,color='k')
off_topspines(ax[0])
off_topspines(ax[1])
plt.savefig("C:\\Users\\Flora\\Pictures\\VEkernels_nl.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
# are there some ephys quality metrics that could distinguish the auditory vs the movement populations 
metrics = cluster_info_test[['VE', 'neuron', 'event',
       'amps', 'depths', 'peakToTrough',
       'amp_max', 'amp_min', 'amp_median', 'amp_std_dB',
       'contamination', 'contamination_alt', 'drift', 'missed_spikes_est',
       'noise_cutoff', 'presence_ratio', 'presence_ratio_std',
       'slidingRP_viol', 'spike_count', 'firing_rate',
       'session_ID']]
metrics = metrics.pivot(index = ['neuron',
       'amps', 'depths', 'peakToTrough',
       'amp_max', 'amp_min', 'amp_median', 'amp_std_dB',
       'contamination', 'contamination_alt', 'drift', 'missed_spikes_est',
       'noise_cutoff', 'presence_ratio', 'presence_ratio_std',
       'slidingRP_viol', 'spike_count', 'firing_rate',
       'session_ID'],columns=['event'])
metrics=metrics.VE
metrics = metrics.reset_index()
metrics['VE,aud-VE,mot'] = metrics.aud-metrics.motionEnergy
thr = .025
metrics= metrics[(metrics.motionEnergy>thr)+(metrics.aud>thr)]
_,ax=plt.subplots(1,1)
sns.pairplot(data=metrics,                  
                  x_vars=['VE,aud-VE,mot'],
                  y_vars=['amps','firing_rate','peakToTrough'])
ax.axvline(0,'k')
plt.savefig("C:\\Users\\Flora\\Pictures\\VEkernels.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
