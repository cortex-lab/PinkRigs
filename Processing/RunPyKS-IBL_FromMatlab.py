from pathlib import Path
import sys,shutil,glob

# general packages for managing paths of data
import pandas as pd 
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 
from pathlib import Path
import numpy as np

# pykilosort 
from pykilosort import run, add_default_handler, Bunch
from pykilosort.params import KilosortParams
import spikeglx 

# PinkRig specific helpers 
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

documents_path = list(Path(r'C:\Users').glob('*\Documents\Github\PinkRigs'))[0].parents[1] # for any user
KS_workpath = documents_path / 'KSworkfolder'

def run_pyKS_single_file(path_to_file, bin_file_extension ='bin'):
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
    print('Starting pyKS-IBL now')

    output_dir = path_to_file.parent / 'pyKS'
    output_dir.mkdir(parents=False, exist_ok=True) 

    #if there is a remainder .kilosort temp processing folder, delete           
    KS_workfolder = KS_workpath / path_to_file.parent.name
    #if KS_workfolder.is_dir():
    #    shutil.rmtree(KS_workfolder)    
    KS_workfolder.mkdir(parents=True, exist_ok=True)     

    add_default_handler(level='INFO') # print output as the algorithm runs
    # find the compressed file of the same name 
    get_bin_string = '*.%s' % bin_file_extension
    print((path_to_file.parent).glob(get_bin_string))
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
    params.channel_detection_method = 'kilosort'
    params.probe = probe
    params = dict(params)

    run(bin_file, dir_path=KS_workfolder, output_dir=output_dir, **params)

    print('DONE')
    success = 1

    # remove the processing products
    #shutil.rmtree(KS_workfolder.joinpath(".kilosort"), ignore_errors=True)

    return success

if __name__ == '__main__':
    bin_file = Path(sys.argv[1])
    success = run_pyKS_single_file(bin_file, bin_file_extension ='bin')
