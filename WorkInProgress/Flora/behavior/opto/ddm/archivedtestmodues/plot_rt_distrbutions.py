#%%

#prepare the data into correct pandas format
import sys
import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,concatenate_events

my_subject = ['AV034','AV025','AV030']
recordings = load_data(
    subject = my_subject,
    expDate = '2021-05-02:2023-09-20',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'table'}}
    )   

ev = concatenate_events(recordings,filter_type='final_stage')
ev.first_choiceDiff = (ev.timeline_firstMoveOn-ev.timeline_choiceMoveOn)
ev.first_choiceDiff[np.isnan(ev.first_choiceDiff)] = 0

ev.rt_toThresh = ev.timeline_choiceThreshOn - ev.timeline_audPeriodOn
ev.first_choiceDiff = (ev.timeline_firstMoveOn-ev.timeline_choiceMoveOn)

ev_ = pd.DataFrame.from_dict(ev)

ev_ = ev_.dropna(subset=['rt'])


# %%

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.hist(ev_.rt_toThresh[ev_.is_visualTrial & ev_.is_validTrial & (ev_.visDiff==.4) & (ev_.response_direction==2) & ~(ev_.first_choiceDiff<0)],bins=100,range=(0,1.5),alpha=.5,color='b')
ax.hist(ev_.rt_toThresh[ev_.is_visualTrial & ev_.is_validTrial & (ev_.visDiff==.4) & (ev_.response_direction==1) & ~(ev_.first_choiceDiff<0)],bins=100,range=(0,1.5),alpha=.5,color='r')



# ax[0].set_ylim([0,8])
# ax[1].set_ylim([8,0])

# %%

t_bins = np.linspace(0,1.5,250)

visContrasts = np.sort(ev_.visDiff.unique())
colors = plt.cm.coolwarm(np.linspace(0,1,visContrasts.size))

for i,c in enumerate(visContrasts):
    pR_t = [np.nanmean(ev_.response_direction[ev_.is_visualTrial & (ev_.visDiff==c) & (ev_.rt_toThresh<t) & ev_.is_validTrial]-1) for t in t_bins]
    #pR_t = [np.mean(ev_.response_direction[ev_.is_visualTrial & (ev_.visDiff==c) & (ev_.rt<t) & ev_.is_validTrial & ~(ev_.first_choiceDiff<0)]-1) for t in t_bins]
    plt.plot(t_bins[1:],pR_t[1:],color=colors[i])

plt.title('visual trials, %s' % my_subject)
plt.xlabel('rt_bin (s)')
plt.ylabel('p(R)')
plt.xlim(0,0.05)
# %%
# same for auditory trials 
audAzimuths = np.sort(ev_.audDiff.unique())

colors = plt.cm.coolwarm(np.linspace(0,1,audAzimuths.size))

for i,c in enumerate(audAzimuths):
    pR_t = [np.mean(ev_.response_direction[ev_.is_auditoryTrial & (ev_.audDiff==c) & (ev_.rt_toThresh<t) & ev_.is_validTrial]-1) for t in t_bins]
    #pR_t = [np.mean(ev_.response_direction[ev_.is_auditoryTrial & (ev_.audDiff==c) & (ev_.rt<t) & ev_.is_validTrial & ~(ev_.first_choiceDiff<0)]-1) for t in t_bins]
    plt.plot(t_bins[3:],pR_t[3:],color=colors[i])


plt.title('auditory trials, %s' % my_subject)
plt.xlabel('rt_bin (s)')
plt.ylabel('p(R)')
# %%
