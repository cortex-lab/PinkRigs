#%%
# load a set of data
# Copy to local folder such that it can be compressed and moved to cluster (easiest is to move the ONE)
# 
import sys,re,shutil
import time
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 
from pathlib import Path
from Admin.csv_queryExp import queryCSV
from Analysis.pyutils.batch_data import get_data_bunch
from Analysis.pyutils.io import save_dict_to_json
dataset = 'naive-allen'
fit_tag = 'additive-fit'

recordings = get_data_bunch(dataset)
unique_ONE_folders = recordings.drop_duplicates(subset=['subject','expDate','expNum'])
target_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\data')
for _,rec_info in unique_ONE_folders.iterrows():
   rec = queryCSV(**rec_info[:3])
   expFolder = rec.iloc[0].expFolder
   cpath  = Path(target_folder.__str__() + re.split('Subjects',expFolder)[-1])
   cpath.mkdir(parents=True,exist_ok=True)
   src = (Path(expFolder) / 'ONE_preproc')
   trgt = cpath/'ONE_preproc'
   if not trgt.is_dir():
    shutil.copytree(src,trgt)
# %%

# write corresponding csv??
ss,ed = [],[]
target_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\data')
target_folder = Path(r'/lustre/home/zcbtfta/data')

for _,rec_info in recordings.iterrows(): 
    rec = queryCSV(**rec_info[:3])
    expFolder = rec.iloc[0].expFolder
    trgt  = Path(target_folder.__str__() + re.split('Subjects',expFolder)[-1])
    ss.append(trgt.__str__())
    ed.append(rec.expDef.values[0])
# %%
recordings['expFolder'] = ss
recordings['expDef'] = ed

recordings.to_csv(target_folder / 'recordings.csv',index=False)
# %%

# contrstruct code structure to take to the cluster
target_folder = Path(r'C:\Users\Flora\Documents\ProcessedData\kernel_regression\code')


# files that we will need
import Admin.csv_queryExp as qE
ct = target_folder / 'Admin'
ct.mkdir(parents=True,exist_ok=True)
shutil.copy(qE.__file__,ct)
shutil.copy(Path(qE.__file__).parent / '__init__.py',ct)


import Analysis
shutil.copytree(Analysis.__path__[0],(target_folder / 'Analysis'),)
# %%
