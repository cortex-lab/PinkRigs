# 
# %%
from pathlib import Path
import shutil 

data_path = Path(r'C:\Users\Flora\Documents\Processed data\ddm\Opto\Data')
animal_paths = list(data_path.glob('*10mW*.csv'))
# %%
new_path = data_path / 'forMyriad'
new_path.mkdir(parents=True,exist_ok=True)
# %%
for p in animal_paths:
    shutil.copy(p,(new_path / p.name))
# %%
