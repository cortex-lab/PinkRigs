"""
running pykilosort on the pyks_queue

"""
# general packages for managing paths of data
import pandas as pd 
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 
from pathlib import Path
from datetime import datetime as time # to sort only for a fixed amount of time


# error handlers 
import sys,shutil,json

# pykilosort 
from pykilosort import run, add_default_handler, neuropixel_probe_from_metafile

def run_pyKS_single_file(path_to_file,recompute_errored_sorting = False):
    """
    try running pyKS on a single file 

    Parameters: 
    path_to_file: pathlib.Path to the ap.bin/ap.cbin or ap.meta files
    recompute_errored_sorting: bool, whether to redo if there was a pyKS_error.json in dir 
    Returns: 
    success: bool, whether the sorting was successful or not. 

    """
    # check if sorting was tried already
    is_tried = (path_to_file.parent / r'pyKS\pyKS_error.json').is_file()
    success = False

    if (is_tried & recompute_errored_sorting) or not is_tried: 
        try:             
            # read the metadata and get the x- and ypos of channels                     

            channel_map  = neuropixel_probe_from_metafile(list((path_to_file.parent).glob('*.meta'))[0] )

            output_dir = path_to_file.parent / 'pyKS'
            output_dir.mkdir(parents=False, exist_ok=True) 

            #if there is a remainder .kilosort temp file, delete 
            temp_file = path_to_file.parent / r'pyKS\.kilosort'
            if temp_file.is_dir():
                shutil.rmtree(temp_file)        

            add_default_handler(level='INFO') # print output as the algorithm runs
            # find the compressed file of the same name 
            input_dir = list((path_to_file.parent).glob('*.cbin'))[0] 
            run(input_dir, probe=channel_map, low_memory=True, dir_path=output_dir)
            # if there was a previous error message, remove it

            err_message_file = path_to_file.parent / r'pyKS\pyKS_error.json'
            
            if err_message_file.is_file(): 
                err_message_file.unlink()
            success=True

        except:
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

            success=False

    return success


def run_pyKS_on_queue(run_for=5.5): 
    """
    run pyKS on the pyKS queue

    Parameters

    run_for: float, how many hrs KS is run for on the queue 
    """

    # get queue
    root = r'\\zserver.cortexlab.net\Code\AVrig\Helpers'
    queue_csv_file = '%s\pykilosort_queue.csv' % root
    queue_csv = pd.read_csv(queue_csv_file)
    print('checking the pyks queue...')

    start_time = time.now()
    start_hour = start_time.hour+start_time.minute/60

    print('current hour is %.2f' % start_hour)

    print('starting my work queue..')
    for idx,rec in queue_csv.iterrows():
        #check if recording is not being sorted already 
        if rec.sortedTag==0: 
            check_time = time.now()
            check_hour = check_time.hour+check_time.minute/60
            if check_hour<(start_hour+run_for): 
                print('still within my time limit... Keep sorting.')                
                input_dir = Path(rec.ephysName)
                queue_csv.sortedTag.iloc[idx]= .5
                queue_csv.to_csv(queue_csv_file,index = False)
                success = run_pyKS_single_file(input_dir,recompute_errored_sorting = True)
                if success:
                    queue_csv.sortedTag.iloc[idx]= 1
                    queue_csv.to_csv(queue_csv_file,index = False) 
                else: 
                    queue_csv.sortedTag.iloc[idx]= -1
                    queue_csv.to_csv(queue_csv_file,index = False) 

if __name__ == "__main__":
   run_pyKS_on_queue(run_for=5.5)
