"""
running pykilosort on the pyks_queue

"""
# general packages for managing paths of data
import pandas as pd 
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 
from pathlib import Path
from datetime import datetime as time # to sort only for a fixed amount of time
from datetime import timedelta
import time as t
# error handlers 
import sys,shutil,glob
import numpy as np

# pykilosort 
from pykilosort import run, add_default_handler, Bunch
from pykilosort.params import KilosortParams
import spikeglx 

# PinkRig specific helpers 
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))
from Admin.csv_queryExp import get_csv_location
#queue updates and ibl formatter 
from Processing.pykilo.check_pyKS_queue import stage_KS_queue
from Processing.pykilo.helpers import save_error_message

documents_path = list(Path(r'C:\Users').glob('*\Documents\Github\PinkRigs'))[0].parents[1] # for any user
KS_workpath = documents_path / 'KSworkfolder'

import pdb
def run_pyKS_single_file(path_to_file,recompute_errored_sorting = False, resort = False, bin_file_extension ='cbin'):
    """
    try running pyKS on a single file 

    Parameters: 
    -------------
    path_to_file: pathlib.Path 
        to the ap.bin/ap.cbin or ap.meta files, 
        implemented this way because sometimes there is both a bin or cbin file ( I think)

    recompute_errored_sorting: bool
        whether to redo if there was a pyKS_error.json in dir 

    
    Returns: 
    ------------
    bool
        whether the sorting was successful or not. 

    """
    # check if sorting was tried with error already
    is_errored = (path_to_file.parent / r'pyKS\pyKS_error.json').is_file()

    # check if sorting was already completed 
    is_sorted = (path_to_file.parent / r'pyKS\output\spike_times.npy').is_file()

    success = False

    # do sorting if errored and asked to recompute error 
    # or not sorted and not errored
    # or sorted, but requested a resort

    if (is_errored & recompute_errored_sorting) or ((not is_sorted) & (not is_errored)) or (is_sorted & resort): 
               
        output_predir = path_to_file.parent / 'pyKS'
        output_predir.mkdir(parents=False, exist_ok=True) 
            
        output_dir = output_predir / 'output'
        output_dir.mkdir(parents=False, exist_ok=True) 

        try:   

            #if there is a remainder .kilosort temp processing folder, delete           
            KS_workfolder = KS_workpath / path_to_file.parent.name
            if KS_workfolder.is_dir():
                shutil.rmtree(KS_workfolder)    
            KS_workfolder.mkdir(parents=True, exist_ok=True)     

            add_default_handler(level='INFO') # print output as the algorithm runs
            # find the compressed file of the same name 
            get_bin_string = '*.%s' % bin_file_extension
            bin_file = list((path_to_file.parent).glob(get_bin_string))[0] 

            # create the probe
            sr = spikeglx.Reader(bin_file)
            h, ver, s2v = (sr.geometry, sr.major_version, sr.sample2volts[0])
            nc = h['x'].size
            probe = Bunch()
            probe.NchanTOT = nc + 1
            probe.chanMap = np.arange(nc)
            probe.xc = h['x'] + h['shank'] * 200
            probe.yc = h['y']
            probe.x = h['x']
            probe.y = h['y']
            probe.shank = h['shank']
            probe.kcoords = np.zeros(nc)
            probe.channel_labels = np.zeros(nc, dtype=int)
            probe.sample_shift = h['sample_shift']
            probe.h, probe.neuropixel_version, probe.sample2volt = (h, ver, s2v)

            # otherwise set default params
            params = KilosortParams()
            params.preprocessing_function = 'destriping'
            params.channel_detection_method = 'raw_correlations'
            params.probe = probe
            params = dict(params)

            run(bin_file, dir_path=KS_workfolder, output_dir=output_dir, **params)

            # if there was a previous error message, remove it
            err_message_file = output_predir / 'pyKS_error.json'          
            if err_message_file.is_file(): 
                err_message_file.unlink()

            # if there was a a previous ibl_format folder, remove it as ibl_format corresponds to the sorting output. 
            ibl_format_path = output_dir / 'ibl_format'
            if ibl_format_path.is_dir(): 
                shutil.rmtree(ibl_format_path)

            success=True

            # remove the processing products
            shutil.rmtree(KS_workfolder.joinpath(".kilosort"), ignore_errors=True)

        except:
            pdb.set_trace()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            save_error_message(output_dir / 'pyKS_error.json',err_type= exc_type,err_message=exc_obj,err_traceback=exc_tb)   
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
    queue_csv_file = get_csv_location('pyKS_queue')
    queue_csv = pd.read_csv(queue_csv_file)
    if overwrite: 
        rec = queue_csv[queue_csv.ephysName==my_ephys_name]
        idx = rec.index.tolist()[0]
        queue_csv.sortedTag.iloc[idx]= overwrite_value
        queue_csv.to_csv(queue_csv_file,index = False)
    return queue_csv
  


def run_pyKS_on_queue(run_for=0.5): 
    """
    run pyKS on the pyKS queue

    Parameters

    run_for: float, how many hrs KS is run for on the queue 
    """
    # update the queue

    
    run_for = float(run_for)
    run_for_minutes = run_for * 60
    print('kilo should be running for %.0f hours' % run_for)
    stage_KS_queue(mouse_selection='allActive',date_selection='previous7',resort=False)   
 
    queue_csv_file = get_csv_location('pyKS_queue')

    start_time = time.now()
    check_time = time.now()-start_time
    # delete the current workpath 
    if KS_workpath.is_dir():
        shutil.rmtree(KS_workpath)

    while check_time<(timedelta(minutes=run_for_minutes)):
        print('current hour is %.2f' % time.now().hour)
        
        print('checking the pyks queue...')
        queue_csv = recheck_queue(overwrite=False)
        to_sort_recs =  queue_csv[queue_csv.sortedTag==0]

        # if there is nothing to sort break the sorting 
        if to_sort_recs.size==0:
            print('appears that there is nothing to sort')
            break
        else: 
            rec = to_sort_recs.iloc[-1] # start from the end of the queue

            _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=.5)
            input_dir = Path(rec.ephysName)            

            success = run_pyKS_single_file(input_dir,recompute_errored_sorting = True)
            if success:
                _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=1)
            else: 
                _ = recheck_queue(overwrite=True,my_ephys_name=rec.ephysName,overwrite_value=-1)
                break

            # update the hour at the end of the loop if still going 
            check_time = time.now()-start_time

if __name__ == "__main__":  
   run_pyKS_on_queue(run_for=14) 
   #run_pyKS_on_queue(run_for=sys.argv[1])
