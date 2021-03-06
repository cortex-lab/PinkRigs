"""
running pykilosort on the pyks_queue

"""
# general packages for managing paths of data
import pandas as pd 
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 
from pathlib import Path
from datetime import datetime as time # to sort only for a fixed amount of time

# error handlers 
import sys,shutil

# pykilosort 
from pykilosort import run, add_default_handler, neuropixel_probe_from_metafile

#queue updates and ibl formatter 
from check_pyKS_queue import stage_KS_queue
from pyhelpers import save_error_message

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
            KS_workfolder = Path(r'C:\Users\Experiment\Documents\KSworkfolder')
            temp_file = KS_workfolder / r'.kilosort'
            if temp_file.is_dir():
                shutil.rmtree(temp_file)        

            add_default_handler(level='INFO') # print output as the algorithm runs
            # find the compressed file of the same name 
            input_dir = list((path_to_file.parent).glob('*.cbin'))[0] 
            run(input_dir, probe=channel_map, low_memory=True, dir_path = KS_workfolder, output_path=output_dir)
            # if there was a previous error message, remove it

            err_message_file = path_to_file.parent / r'pyKS\pyKS_error.json'
            
            if err_message_file.is_file(): 
                err_message_file.unlink()
            success=True

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            save_error_message(output_dir / 'pyKS_error.json',err_type= exc_type,err_message=exc_obj)   

            success=False

    return success

def recheck_queue(overwrite=True,my_ephys_name='None',overwrite_value=1): 
    """
    function that loads the pyKS queue 
    optionally overwrites specific rows defined by my_ephys_name

    Parameters:
    ----------------
    overwrite: bool 
    my_ephys_name: str
    overwrite_value: float 
    
    """

    root = r'\\zserver.cortexlab.net\Code\AVrig\Helpers'
    queue_csv_file = '%s\pykilosort_queue.csv' % root
    queue_csv = pd.read_csv(queue_csv_file)
    if overwrite: 
        rec = queue_csv[queue_csv.ephysName==my_ephys_name]
        idx = rec.index.tolist()[0]
        queue_csv.sortedTag.iloc[idx]= overwrite_value
        queue_csv.to_csv(queue_csv_file,index = False)
    return queue_csv
  


def run_pyKS_on_queue(run_for=5.5): 
    """
    run pyKS on the pyKS queue

    Parameters

    run_for: float, how many hrs KS is run for on the queue 
    """
    # update the queue

    run_for = float(run_for)
    stage_KS_queue(mouse_selection='allActive',date_selection='last7')    

    root = r'\\zserver.cortexlab.net\Code\AVrig\Helpers'
    queue_csv_file = '%s\pykilosort_queue.csv' % root

    start_time = time.now()
    start_hour = start_time.hour+start_time.minute/60
    check_hour = start_hour

    while check_hour<(start_hour+run_for):
        print('current hour is %.2f' % start_hour)
        
        print('checking the pyks queue...')
        queue_csv = recheck_queue(overwrite=False)
        to_sort_recs =  queue_csv[queue_csv.sortedTag==0]

        # if there is nothing to sort break the sorting 
        if to_sort_recs.size==0:
            print('appears that there is nothing to sort')
            break
        else: 
            rec = to_sort_recs.iloc[0]

            _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=.5)
            input_dir = Path(rec.ephysName)
            success = run_pyKS_single_file(input_dir,recompute_errored_sorting = True)
            if success:
                _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=1)
            else: 
                _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=-1)

            # update the hour at the end of the loop if still going 
            check_time = time.now()
            check_hour = check_time.hour+check_time.minute/60

if __name__ == "__main__":  
    run_pyKS_on_queue() 
   #run_pyKS_on_queue(run_for=sys.argv[1])
