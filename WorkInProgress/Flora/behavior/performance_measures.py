# %%
import sys
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Admin.csv_queryExp import load_data,simplify_recdat,Bunch
from Analysis.pyutils.plotting import off_topspines
from Analysis.pyutils.ev_dat import getTrialNames

my_subject = ['AV038','AV036','AV031','AV029','AV033','AV034','AV035','AV030']
recordings = load_data(
    subject = my_subject,
    expDate = 'last15',
    expDef = 'multiSpaceWorld_checker_training',
    checkEvents = '1', 
    data_name_dict={'events':{'_av_trials':'all'}}
    )

# %%
pL,pR = [],[]
rigs = []
subject = []
for rig in recordings.rigName.unique():
    for s in my_subject: 
        rec_a = recordings[(recordings.subject==s) & (recordings.rigName==rig)]
        for _,rec in rec_a.iterrows():
            
            ev = rec.events._av_trials
            to_keep_trials = ev.is_auditoryTrial & ev.is_validTrial & ~np.isnan(ev.timeline_choiceMoveDir) & ~(ev.is_laserTrial==1)
            ev_  = Bunch({k:ev[k][to_keep_trials] for k in ev.keys()})



            aud_azimuths = [-60,60]
            aud_perfs = np.array([np.mean((ev_.response_feedback[ev_.stim_audAzimuth==a]+1)/2) for a in aud_azimuths])

            pL.append(aud_perfs[0])
            pR.append(aud_perfs[1])
            rigs.append(rig)
            subject.append(s)
        
mydict = {'pR':pR,'pL':pL,'rigName':rigs,'mouse':subject}
        

# %%
import pandas as pd
df = pd.DataFrame(mydict)
df['audR-audL'] = df.pR-df.pL
import seaborn as sns
_,ax = plt.subplots(1,1,figsize=(10,10))
sns.boxenplot(df,x='rigName',y='audR-audL',ax=ax)
# %%
