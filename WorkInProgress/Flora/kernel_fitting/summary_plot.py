# %%
from pathlib import Path

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
# %
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from Analysis.pyutils.plotting import off_topspines,off_axes

p  = Path(r'C:\Users\Flora\Documents\Processed data\Audiovisual\naive-allen\kernel_model\additive-nl-comp') 
res = list(p.glob('*.csv'))

df = pd.concat([pd.read_csv(r) for r in res])

plt.rcParams.update({'font.family':'Calibri'})
plt.rcParams.update({'font.size':28})
fig,ax = plt.subplots(1,1,figsize=(5,5))
sns.scatterplot(data=df[df.additive.values>-1], x="additive", y="gain",ax=ax,s=50)
ax.plot([-.2,.8],[-.2,.8],'k--')
#ax.plot([0,.1],[0,.0],'k--')
#ax.plot([0,.0],[0,.1],'k--')
# ax.set_xlim([0,0.1])
# ax.set_ylim([0,0.1])

off_topspines(ax)
fig.savefig("C:\\Users\\Flora\\Pictures\\LakeConf\\nl-additive.svg",transparent=False,bbox_inches = "tight",format='svg',dpi=300)

# %%
