# %%
import pandas as pd 
import numpy as np 
import itertools
ev = pd.read_csv(r'\\znas.cortexlab.net\Lab\Share\Flora\forMax\lowSPLmice_ctrl.csv') # batch data

ev = ev[ev.subject=='AV034']
ev = ev[ev.is_validTrial]

vis = np.sort(np.unique(ev.visDiff))    
aud = np.sort(np.unique(ev.audDiff))

for a,v in itertools.product(aud,vis): 
    ev.rt_thresh[ev.response_feedback==1]

print('idk why my thing is not working')
# %%

# %%
