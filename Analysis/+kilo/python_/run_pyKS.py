"""
running pykilosort on the pyks_queue

"""
# general packages for managing paths of data
import pandas as pd 
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 
from pathlib import Path

# error handlers 
import sys
import json

# pykilosort 
from pykilosort import run, add_default_handler, neuropixel_probe_from_metafile


# get queue
root = r'\\zserver.cortexlab.net\Code\AVrig\Helpers'
queue_csv_file = '%s\pykilosort_queue.csv' % root
queue_csv = pd.read_csv(queue_csv_file)
print('checking the pyks queue...')
for idx,rec in queue_csv.iterrows():
    #check if recording is not being sorted already 
    if rec.sortedTag==0: 

        try:
            input_dir = Path(rec.ephysName)     
            print(input_dir)       
            # read the metadata and get the x- and ypos of channels 
            channel_map  = neuropixel_probe_from_metafile(list((input_dir.parents[0]).glob('*.meta'))[0])

            output_dir = input_dir.parents[0] / 'pyKS'
            output_dir.mkdir(parents=False, exist_ok=True)
            
            queue_csv.sortedTag.iloc[idx]= .5
            queue_csv.to_csv(queue_csv_file,index = False)
            add_default_handler(level='INFO') # print output as the algorithm runs
            # find the compressed file of the same name 
            input_dir = list((input_dir.parents[0]).glob('*.cbin'))[0] 
            run(input_dir, probe=channel_map, low_memory=True, dir_path=output_dir)
            queue_csv.sortedTag.iloc[idx]= 1
            queue_csv.to_csv(queue_csv_file,index = False)
        except: 
            # save error message 
            exc_type, exc_obj, exc_tb = sys.exc_info()
    
            errdict = {
                'err_type:': str(exc_type), 
                'err_message': str(exc_obj),
                'traceback': str(exc_tb)
                }

            errmessage = json.dumps(errdict)
            
            errfile = open(output_dir / 'pyKS_error.json',"w")
            errfile.write(errmessage)
            errfile.close()

            # delete temp folders before Michael kicks me out

            # update csv                
            queue_csv.sortedTag.iloc[idx]= -1
            queue_csv.to_csv(queue_csv_file,index = False)  