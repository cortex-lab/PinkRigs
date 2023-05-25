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
destination = r"C:\Users\Flora\Documents\testcam"

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
