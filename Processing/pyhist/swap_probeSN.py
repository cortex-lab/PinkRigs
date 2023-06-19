# %%
# this function is meant to swap probe serial numbers
import shutil,re
from pathlib import Path
import numpy as np

my_folder = r'Z:\AV008\histology\registration\brainreg_output\manual_segmentation\standard_space\tracks'
my_folder = Path(my_folder)
swap_SN = False  
swap_order = True
files = list(my_folder.glob('*.*'))

SNs = [re.findall('SN(.*?)_',f.name)[0] for f in files]

uniqueSNs = np.unique(SNs)

# %%

from datetime import datetime
now = datetime.now()
timestring = now.strftime("%Y-%m-%d_%H_%M")
folder_name = 'pre_swapping_' + timestring
# create a folder pre_swapping an
old_file_path = my_folder / folder_name
old_file_path.mkdir(parents=False, exist_ok=True) 
# copy all files to that folder 
# %%
if swap_SN: 
    new_filenames = []
    for f in files: 
        dest = old_file_path / f.name
        shutil.copy(f, dest)

        curr_sn = re.findall('SN(.*?)_',f.name)[0] 
        newSN_idx = np.where(uniqueSNs!=curr_sn)[0][0]
        new_SN = uniqueSNs[newSN_idx]
        new_name = f.name.replace(curr_sn,new_SN)
        new_filenames.append(new_name)

    for f in files: 
        f.unlink()
    
    files_ = list(old_file_path.glob('*'))


    for f,n in zip(files_,new_filenames):
        dest = my_folder / n
        shutil.copy(f, dest)


# %%

if swap_order & (not swap_SN):
    new_filenames = []
    for f in files: 
        dest = old_file_path / f.name
        shutil.copy(f, dest)

        new_name = f.stem[:-1] + str(3-int(f.stem[-1])) + f.suffix
        new_filenames.append(new_name)

    for f in files: 
        f.unlink()
    
    files_ = list(old_file_path.glob('*'))


    for f,n in zip(files_,new_filenames):
        dest = my_folder / n
        shutil.copy(f, dest)


# %%
