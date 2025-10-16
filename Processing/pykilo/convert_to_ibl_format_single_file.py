# general packages for managing paths of data
import glob,sys
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None # disable warning, we will overwrite some rows when sortedTag changes 

# ibl ephys tools 
import spikeglx
from atlaselectrophysiology.extract_files import ks2_to_alf,extract_rmsmap,_sample2v
from ibllib.atlas import AllenAtlas
atlas = AllenAtlas(25) # always register to the 25 um atlas 

# PinkRig processing tools 
# import PinkRig ephys processng tools
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

def extract_data_PinkRigs(ks_path, ephys_path, out_path,do_raw_files=False):
    efiles = spikeglx.glob_ephys_files(ephys_path)

    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            ks2_to_alf(ks_path, ephys_path, out_path, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=True)

            # I might need to rewrite the channels.localCoordinate files. These seem to be wrong
            # on npix 2.0. (as in: incorrect spcing and positional identifier.)
            if do_raw_files:
                extract_rmsmap(efile.ap, out_folder=out_path, spectra=False)
        if efile.get('lf') and efile.lf.exists() and do_raw_files:
            extract_rmsmap(efile.lf, out_folder=out_path)

def ks_to_ibl_format(ephys_path,ks_folder='kilosort4',recompute=False):
        
    print(ephys_path)
    ks_path = ephys_path / ks_folder
    # Save path
    out_path = ks_path / 'ibl_format'
    out_path.mkdir(parents=False, exist_ok=True) # make directory if it does not exist

    # extract the data to ibl_format if it has not been done already.
    print(out_path)
    if not (out_path / 'cluster_metrics.csv').is_file() or recompute:
        print('converting data to IBL format ...')
        extract_data_PinkRigs(ks_path, ephys_path, out_path,do_raw_files=False) 

    return out_path

if __name__ == "__main__":
   ephys_path = Path(sys.argv[1])
   ks_folder = sys.argv[2]
   ks_to_ibl_format(ephys_path,ks_folder=ks_folder)