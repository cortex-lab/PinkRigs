# %%
import glob,sys,shutil
from pathlib import Path 

pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Admin.csv_queryExp import queryCSV

recordings = queryCSV(subject = 'all',expDate='last1',expDef='multiSpaceWorld_checker_training')
recordings = recordings[recordings.existSideCam=='1']

# Source path
destination = Path(r"C:\Users\Flora\Documents\testcam")

# %%
for _,rec in recordings.iterrows():
    camfilename = '%s_%s_%s_sideCam.mj2' % (rec.expDate,rec.expNum,rec.subject)

    source = Path(rec.expFolder) 
    source = source / camfilename
    # Copy the content of
    # source to destination
    
    try:
        shutil.copy(source, destination)
        print(" %s copied successfully." % camfilename)
    
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    
    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")
    
    # For other errors
    except:
        print("Error occurred while copying %s." % camfilename)


# %%

from distutils.dir_util import copy_tree
for _,rec in recordings.iterrows():

    my_expFolder = Path(rec.expFolder)
    folders = my_expFolder.parts[1:]

    from_directory = my_expFolder / 'ONE_preproc'
    to_directory = destination / ('%s/%s/%s' % (folders[0],folders[1],folders[2]))
    to_directory = to_directory / 'ONE_preproc'

    to_directory.mkdir(parents=True,exist_ok=True)

    copy_tree(from_directory.__str__(), to_directory.__str__())
# %%
