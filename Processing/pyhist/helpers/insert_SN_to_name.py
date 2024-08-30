# %%
import re
from pathlib import Path
subject = 'AV009'

path  = r'\\zinu.cortexlab.net\Subjects\%s' % subject
histpath  = (r'\histology\registration\brainreg_output' + 
             r'\manual_segmentation\standard_space\tracks')

my_path = Path(path  + histpath)

files = list(my_path.glob('*'))

for f in files:
    pattern = re.compile(r'(AV\d+_)(\d+)(_shank\d+)')
    if f.is_file():
        filename = f.name
        extension = f.suffix
        match = pattern.match(filename)
        if match:
            new_filename = f"{match.group(1)}SN{match.group(2)}{match.group(3)}{extension}"            
            # Create the new filepath
            new_filepath = f.with_name(new_filename)
            f.rename(new_filepath)
        
# %%
